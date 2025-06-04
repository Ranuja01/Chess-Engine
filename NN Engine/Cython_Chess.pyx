# -*- coding: utf-8 -*-
"""
@author: Ranuja Pinnaduwage

This file contains cythonized code to emulate the python-chess components for generating legal moves

Code augmented from python-chess: https://github.com/niklasf/python-chess/tree/5826ef5dd1c463654d2479408a7ddf56a91603d6

"""

from libc.stdlib cimport rand, srand, RAND_MAX, malloc, free
from libc.math cimport fmod
from libc.time cimport time
from time import time as py_time 
cimport cython
import chess
import itertools
import random
from collections.abc import Iterator


# From the c++ standard library import vectors and strings
from libcpp.vector cimport vector
from libcpp.string cimport string

# Define C data types 
cdef extern from "stdint.h":
    ctypedef signed char int8_t
    ctypedef unsigned char uint8_t
    ctypedef unsigned long long uint64_t

# Import functions from the c++ file
cdef extern from "cpp_bitboard.h":
    void process_bitboards_wrapper(uint64_t * bitboards, int size)
    vector[int] find_most_significant_bits(uint64_t bitmask)  
    uint8_t scan_reversed_size(uint64_t bb)
    void scan_reversed(uint64_t bb, vector[uint8_t] &result)
    void scan_forward(uint64_t bb, vector[uint8_t] &result)
    vector[uint8_t] scan_reversedOld(uint64_t bb)
    int getPPIncrement(int square, bint colour, uint64_t opposingPawnMask, int ppIncrement, int x)
    uint64_t attacks_mask(bint colour, uint64_t occupied, uint8_t square, uint8_t pieceType)
    uint64_t attackersMask(bint colour, uint8_t square, uint64_t occupied, uint64_t queens_and_rooks, uint64_t queens_and_bishops, uint64_t kings, uint64_t knights, uint64_t pawns, uint64_t occupied_co)
    uint64_t slider_blockers(uint8_t king, uint64_t queens_and_rooks, uint64_t queens_and_bishops, uint64_t occupied_co_opp, uint64_t occupied_co, uint64_t occupied)
    uint64_t betweenPieces(uint8_t a, uint8_t b)
    uint64_t ray(uint8_t a, uint8_t b)
    void update_bitmasks(uint64_t pawnsMask, uint64_t knightsMask, uint64_t bishopsMask, uint64_t rooksMask, uint64_t queensMask, uint64_t kingsMask, uint64_t occupied_whiteMask, uint64_t occupied_blackMask, uint64_t occupiedMask)
    bint is_safe(uint8_t king, uint64_t blockers, uint8_t from_square, uint8_t to_square, uint64_t occupiedMask, uint64_t occupiedWhite, uint64_t opposingPieces, uint64_t ourPieces, uint64_t pawnsMask, uint64_t knightsMask, uint64_t bishopsMask,
			 uint64_t rooksMask, uint64_t queensMask, uint64_t kingsMask, int ep_square, bint turn)
    
    bint is_capture(uint8_t from_square, uint8_t to_square, uint64_t occupied_co, bint is_en_passant)
    bint is_check(bint colour, uint64_t occupied, uint64_t queens_and_rooks, uint64_t queens_and_bishops, uint64_t kings, uint64_t knights, uint64_t pawns, uint64_t opposingPieces)
    void initialize_attack_tables()
    void setAttackingLayer(int increment);
    void printLayers();
    
    void generateLegalMoves(vector[uint8_t] &startPos_filtered, vector[uint8_t] &endPos_filtered, vector[uint8_t] &promotions_filtered,  uint64_t preliminary_castling_mask, uint64_t from_mask, uint64_t to_mask,
	 					uint64_t occupiedMask, uint64_t occupiedWhite, uint64_t opposingPieces, uint64_t ourPieces, uint64_t pawnsMask, uint64_t knightsMask, uint64_t bishopsMask,
						uint64_t rooksMask, uint64_t queensMask, uint64_t kingsMask, int ep_square, bint turn);
    
    void generatePseudoLegalMoves(vector[uint8_t] &startPos, vector[uint8_t] &endPos, vector[uint8_t] &promotions, uint64_t preliminary_castling_mask, uint64_t from_mask, uint64_t to_mask, uint64_t king,  uint64_t occupiedMask, uint64_t occupiedWhite, uint64_t opposingPieces, uint64_t ourPieces, uint64_t pawnsMask, uint64_t knightsMask, uint64_t bishopsMask,
							      uint64_t rooksMask, uint64_t queensMask, uint64_t kingsMask, int ep_square, bint turn)
    
    void generatePieceMoves(vector[uint8_t] &startPos, vector[uint8_t] &endPos, vector[uint8_t] &promotions, uint64_t our_pieces, uint64_t from_mask, uint64_t to_mask, uint64_t occupiedMask,
	 					    uint64_t occupiedWhite, uint64_t pawnsMask, uint64_t knightsMask, uint64_t bishopsMask, uint64_t rooksMask, uint64_t queensMask, uint64_t kingsMask)
    
    void generatePawnMoves(vector[uint8_t] &startPos, vector[uint8_t] &endPos, vector[uint8_t] &promotions, uint64_t opposingPieces,
                           bint colour, uint64_t pawnsMask, uint64_t occupiedMask, uint64_t from_mask, uint64_t to_mask)
    
    void generateCastlingMoves(vector[uint8_t] &startPos, vector[uint8_t] &endPos, vector[uint8_t] &promotions, uint64_t preliminary_castling_mask, uint64_t to_mask, uint64_t king, 
                               uint64_t ourPieces, uint64_t occupiedMask, uint64_t pawnsMask, uint64_t knightsMask, uint64_t bishopsMask, uint64_t rooksMask, uint64_t queensMask, uint64_t kingsMask, bint turn)
    
    void generateEnPassentMoves(vector[uint8_t] &startPos, vector[uint8_t] &endPos, vector[uint8_t] &promotions, uint64_t from_mask, uint64_t to_mask,
                                uint64_t our_pieces, uint64_t occupiedMask, uint64_t pawnsMask, int ep_square, bint turn)
    
    void generateEvasions(vector[uint8_t] &startPos, vector[uint8_t] &endPos, vector[uint8_t] &promotions, uint64_t preliminary_castling_mask, uint8_t king, uint64_t checkers, uint64_t from_mask, uint64_t to_mask, uint64_t occupiedMask, uint64_t occupiedWhite, uint64_t opposingPieces, uint64_t ourPieces,
					  uint64_t pawnsMask, uint64_t knightsMask, uint64_t bishopsMask, uint64_t rooksMask, uint64_t queensMask, uint64_t kingsMask, int ep_square, bint turn)
    
    void generateLegalCaptures(vector[uint8_t] &startPos_filtered, vector[uint8_t] &endPos_filtered, vector[uint8_t] &promotions_filtered, uint64_t from_mask, uint64_t to_mask,
	 					uint64_t occupiedMask, uint64_t occupiedWhite, uint64_t opposingPieces, uint64_t ourPieces, uint64_t pawnsMask, uint64_t knightsMask, uint64_t bishopsMask,
						uint64_t rooksMask, uint64_t queensMask, uint64_t kingsMask, int ep_square, bint turn)
    
    void generateLegalMovesReordered(vector[uint8_t] &startPos, vector[uint8_t] &endPos, vector[uint8_t] &promotions, uint64_t preliminary_castling_mask, uint64_t from_mask, uint64_t to_mask,
								 uint64_t occupiedMask, uint64_t occupiedWhite, uint64_t opposingPieces, uint64_t ourPieces, uint64_t pawnsMask, uint64_t knightsMask,
								 uint64_t bishopsMask, uint64_t rooksMask, uint64_t queensMask, uint64_t kingsMask, int ep_square, bint turn)
    
    void initializeZobrist()
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
    
    int remove_piece_at(uint8_t square, uint64_t& pawnsMask, uint64_t& knightsMask, uint64_t& bishopsMask, uint64_t& rooksMask, uint64_t& queensMask, uint64_t& kingsMask, uint64_t& occupiedMask, uint64_t& occupiedWhite, uint64_t& occupiedBlack, uint64_t& promoted)
    void set_piece_at(uint8_t square, uint8_t piece_type, uint64_t& pawnsMask, uint64_t& knightsMask, uint64_t& bishopsMask, uint64_t& rooksMask, uint64_t& queensMask, uint64_t& kingsMask, uint64_t& occupiedMask, uint64_t& occupiedWhite, uint64_t& occupiedBlack, uint64_t& promoted, bint promotedFlag, bint turn)
    void update_state(uint8_t to_square, uint8_t from_square, uint64_t& pawnsMask, uint64_t& knightsMask, uint64_t& bishopsMask, uint64_t& rooksMask, uint64_t& queensMask, uint64_t& kingsMask, uint64_t& occupiedMask, uint64_t& occupiedWhite, uint64_t& occupiedBlack, uint64_t& promoted, uint64_t& castling_rights, int& ep_square, int promotion_type, bint turn)


cdef struct Bitboards:
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
    uint64_t castling_rights
    int ep_square

cdef void fill_bitboards(Bitboards* bb, object board):
    bb.pawns = board.pawns
    bb.knights = board.knights
    bb.bishops = board.bishops
    bb.rooks = board.rooks
    bb.queens = board.queens
    bb.kings = board.kings
    bb.occupied_white = board.occupied_co[True]
    bb.occupied_black = board.occupied_co[False]
    bb.occupied = board.occupied
    bb.promoted = board.promoted
    bb.castling_rights = board.castling_rights
    
    if (board.ep_square):
        bb.ep_square = board.ep_square
    else:
        bb.ep_square = -1
    

cdef void copy_back(Bitboards* bb, object board):
    board.pawns = bb.pawns
    board.knights = bb.knights
    board.bishops = bb.bishops
    board.rooks = bb.rooks
    board.queens = bb.queens
    board.kings = bb.kings
    board.occupied_co[True] = bb.occupied_white
    board.occupied_co[False] = bb.occupied_black
    board.occupied = bb.occupied
    board.promoted = bb.promoted
    board.castling_rights = bb.castling_rights
    
    if (bb.ep_square == -1):
        board.ep_square = None
    else:
        board.ep_square = bb.ep_square
        


def generate_legal_moves(object board, uint64_t from_mask, uint64_t to_mask) -> Iterator[chess.Move]:

    cdef vector[uint8_t] startPos_filtered
    cdef vector[uint8_t] endPos_filtered
    cdef vector[uint8_t] promotions_filtered
    
    startPos_filtered.reserve(128)
    endPos_filtered.reserve(128)
    promotions_filtered.reserve(128)
    
    cdef uint8_t num_moves
    
    cdef uint64_t preliminary_castling_mask = 0
    
    if from_mask & board.kings:
        preliminary_castling_mask = board.clean_castling_rights()

    # update_bitmasks(board.pawns,board.knights,board.bishops,board.rooks,board.queens,board.kings,board.occupied_co[True],board.occupied_co[False],board.occupied)

    generateLegalMoves(startPos_filtered, endPos_filtered, promotions_filtered, preliminary_castling_mask, from_mask, to_mask, board.occupied, board.occupied_co[True],
                       board.occupied_co[not board.turn], board.occupied_co[board.turn], board.pawns, board.knights, board.bishops,
					   board.rooks, board.queens, board.kings, (-1 if board.ep_square is None else board.ep_square), board.turn);   
    
    num_moves = startPos_filtered.size()  
    for i in range(num_moves):
        if (promotions_filtered[i] == 1):
            yield chess.Move(startPos_filtered[i], endPos_filtered[i])
        else:
            yield chess.Move(startPos_filtered[i], endPos_filtered[i], promotions_filtered[i])
    

def generate_legal_captures(object board, uint64_t from_mask, uint64_t to_mask) -> Iterator[chess.Move]:
    
    """
    Function to generate legal captures through yielding

    Parameters:
    - board: The current board state
    - from_mask: The starting position mask
    - to_mask: The ending position mask

    Yields:
    - Legal capture chess.Moves
    """
    
    # Use itertools to yield moves from the legal moves function and en passent function        
    # return itertools.chain(
    #     generate_legal_moves(board,from_mask, to_mask & board.occupied_co[not board.turn]),
    #     generate_legal_ep(board,from_mask, to_mask))   

    cdef vector[uint8_t] startPos_filtered
    cdef vector[uint8_t] endPos_filtered
    cdef vector[uint8_t] promotions_filtered
    
    startPos_filtered.reserve(128)
    endPos_filtered.reserve(128)
    promotions_filtered.reserve(128)
    
    cdef uint8_t num_moves
        
    generateLegalCaptures(startPos_filtered, endPos_filtered, promotions_filtered, from_mask, to_mask, board.occupied, board.occupied_co[True], board.occupied_co[not board.turn],
                          board.occupied_co[board.turn], board.pawns, board.knights,board.bishops, board.rooks, board.queens, board.kings,
                          (-1 if board.ep_square is None else board.ep_square), board.turn)

    num_moves = startPos_filtered.size()  
    for i in range(num_moves):
        if (promotions_filtered[i] == 1):
            yield chess.Move(startPos_filtered[i], endPos_filtered[i])
        else:
            yield chess.Move(startPos_filtered[i], endPos_filtered[i], promotions_filtered[i])     
            
def generate_ordered_moves(object board, uint64_t from_mask, uint64_t to_mask):
    cdef vector[uint8_t] startPos
    cdef vector[uint8_t] endPos
    cdef vector[uint8_t] promotions
    
    startPos.reserve(128)
    endPos.reserve(128)
    promotions.reserve(128)
    
    cdef uint8_t num_moves
    
    cdef uint64_t preliminary_castling_mask = 0
    
    if from_mask & board.kings:
        preliminary_castling_mask = board.clean_castling_rights()

    generateLegalMovesReordered(startPos, endPos, promotions, preliminary_castling_mask, from_mask, to_mask, board.occupied, board.occupied_co[True], board.occupied_co[not board.turn],
                          board.occupied_co[board.turn], board.pawns, board.knights,board.bishops, board.rooks, board.queens, board.kings,
                          (-1 if board.ep_square is None else board.ep_square), board.turn)
    
    num_moves = startPos.size()  
    for i in range(num_moves):
        if (promotions[i] == 1):
            yield chess.Move(startPos[i], endPos[i])
        else:
            yield chess.Move(startPos[i], endPos[i], promotions[i])     

def generate_pseudo_legal_moves(object board, uint64_t from_mask, uint64_t to_mask, uint64_t king) -> Iterator[chess.Move]:
    
    """
    Function to generate pseudo legal moves through yielding

    Parameters:
    - board: The current board state
    - from_mask: The starting position mask
    - to_mask: The ending position mask

    Yields:
    - Legal evasion chess.Moves
    """
        
    cdef vector[uint8_t] startPos
    cdef vector[uint8_t] endPos
    cdef vector[uint8_t] promotions
    
    cdef uint8_t num_moves
    
    cdef uint64_t preliminary_castling_mask = 0
    
    if from_mask & board.kings:
        preliminary_castling_mask = board.clean_castling_rights()
    
    generatePseudoLegalMoves(startPos, endPos, promotions, preliminary_castling_mask, from_mask, to_mask, king,
                             board.occupied, board.occupied_co[True], board.occupied_co[not board.turn], board.occupied_co[board.turn], board.pawns, board.knights,
                             board.bishops, board.rooks, board.queens, board.kings, (-1 if board.ep_square is None else board.ep_square), board.turn)

    num_moves = startPos.size()  
    for i in range(num_moves):
        if (promotions[i] == 1):
            yield chess.Move(startPos[i], endPos[i])
        else:
            yield chess.Move(startPos[i], endPos[i], promotions[i])

def push(object board, object move):
    
    # Push move and remember board state.
       
    cdef Bitboards bb
        
    board_state = chess._BoardState(board)
    board.castling_rights = board.clean_castling_rights()  # Before pushing stack
    board.move_stack.append(move)
    board._stack.append(board_state)

    # Reset en passant square.
    ep_square = board.ep_square
    board.ep_square = None

    # Increment move counters.
    board.halfmove_clock += 1
    if board.turn == chess.BLACK:
        board.fullmove_number += 1

    # On a null move, simply swap turns and reset the en passant square.
    if not move:
        board.turn = not board.turn
        return

    # Zero the half-move clock.
    if board.is_zeroing(move):
        board.halfmove_clock = 0
        
    fill_bitboards(&bb, board)
    
    update_state(move.to_square, move.from_square, bb.pawns, bb.knights, bb.bishops,
                 bb.rooks, bb.queens, bb.kings,
                 bb.occupied, bb.occupied_white, bb.occupied_black,
                 bb.promoted, bb.castling_rights, bb.ep_square, (move.promotion if move.promotion else 1), board.turn)
    
    # Swap turn.
    board.turn = not board.turn
    copy_back(&bb, board)

def gives_check(object board,object move):
    board.push(move)
    
    try:
        return is_check(board.turn,board.occupied, board.queens | board.rooks, board.queens | board.bishops, board.kings, board.knights, board.pawns, board.occupied_co[not board.turn])    
    finally:
        board.pop()
    
def inititalize():
    initializeZobrist()
    initialize_attack_tables()