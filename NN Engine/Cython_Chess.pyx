# cython_sum.pyx

from libc.stdlib cimport rand, srand, RAND_MAX, malloc, free
from libc.math cimport fmod
from libc.time cimport time
from time import time as py_time 
cimport cython
import chess
import itertools
import random
from typing import Iterator

# Cython_Chess.pyx
from libcpp.vector cimport vector
from libcpp.string cimport string
cdef extern from "stdint.h":
    ctypedef signed char int8_t
    ctypedef unsigned char uint8_t
    ctypedef unsigned long long uint64_t
    
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
    bint is_capture(uint8_t from_square, uint8_t to_square, uint64_t occupied_co, bint is_en_passant)
    bint is_check(bint colour, uint64_t occupied, uint64_t queens_and_rooks, uint64_t queens_and_bishops, uint64_t kings, uint64_t knights, uint64_t pawns, uint64_t opposingPieces)
    void initialize_attack_tables()
    void setAttackingLayer(uint64_t occupied_white, uint64_t occupied_black, uint64_t kings, int increment);
    void printLayers();
    void generatePieceMoves(vector[uint8_t] &startPos, vector[uint8_t] &endPos, uint64_t our_pieces, uint64_t pawnsMask, uint64_t knightsMask, uint64_t bishopsMask, uint64_t rooksMask, uint64_t queensMask, uint64_t kingsMask, uint64_t occupied_whiteMask, uint64_t occupied_blackMask, uint64_t occupiedMask, uint64_t from_mask, uint64_t to_mask)
    void generatePawnMoves(vector[uint8_t] &startPos, vector[uint8_t] &endPos, vector[uint8_t] &promotions, uint64_t opposingPieces, uint64_t occupied, bint colour, uint64_t pawnsMask, uint64_t from_mask, uint64_t to_mask)
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
    
    
cdef int layer[2][8][8]
# Initialize layer
# Initialize layer
for i in range(2):
    for j in range(8):
        for k in range(8):
            if i == 0:
                layer[i][j][k] = [
                    [0,0,0,0,0,0,0,0],
                    [0,0,3,3,4,5,5,0],
                    [0,0,3,6,7,6,4,0],
                    [0,0,3,7,8,8,5,0],
                    [0,0,3,7,8,8,5,0],
                    [0,0,3,6,7,6,4,0],
                    [0,0,3,3,4,5,5,0],
                    [0,0,0,0,0,0,0,0]
                ][j][k]
            else:
                layer[i][j][k] = [
                    [0,0,0,0,0,0,0,0],
                    [0,5,5,4,3,3,0,0],
                    [0,4,6,7,6,3,0,0],
                    [0,5,8,8,7,3,0,0],
                    [0,5,8,8,7,3,0,0],
                    [0,4,6,7,6,3,0,0],
                    [0,5,5,4,3,3,0,0],
                    [0,0,0,0,0,0,0,0]
                ][j][k]

def generate_legal_moves(object board, uint64_t from_mask, uint64_t to_mask) -> Iterator[chess.Move]:
    
    cdef uint8_t king
    cdef uint64_t blockers
    cdef uint64_t checkers
    cdef object move
    
    if from_mask is None:
        from_mask = chess.BB_ALL
    if to_mask is None:
        to_mask = chess.BB_ALL
    
    if board.is_variant_end():
        return

    cdef uint64_t king_mask = board.kings & board.occupied_co[board.turn]
    if king_mask:
        #king = chess.msb(king_mask)
        king = king_mask.bit_length() - 1
        blockers = slider_blockers(king, board.queens | board.rooks, board.queens | board.bishops, board.occupied_co[not board.turn], board.occupied_co[board.turn], board.occupied)
        #blockers = _slider_blockers(board,king)
        #blockers = board._slider_blockers(king)
        
        checkers = attackersMask(not board.turn, king, board.occupied, board.queens | board.rooks, board.queens | board.bishops, board.kings, board.knights, board.pawns, board.occupied_co[not board.turn])
        #checkers = attackers_mask(board, not board.turn, king,board.occupied)
        #checkers = board.attackers_mask(not board.turn, king)
        if checkers:
            for move in _generate_evasions(board, king, checkers, from_mask, to_mask):
                if board._is_safe(king, blockers, move):
                    yield move
        else:
            for move in generate_pseudo_legal_moves(board, from_mask, to_mask):
            #for move in board.generate_pseudo_legal_moves(from_mask, to_mask):
                if board._is_safe(king, blockers, move):
                    yield move
    else:
        yield from generate_pseudo_legal_moves(board, from_mask, to_mask)

def generate_legal_ep(object board, uint64_t from_mask, uint64_t to_mask) -> Iterator[chess.Move]:
    
    cdef object move
    if board.is_variant_end():
        return

    for move in board.generate_pseudo_legal_ep(from_mask, to_mask):
        if not board.is_into_check(move):
            yield move

def generate_legal_captures(object board, uint64_t from_mask, uint64_t to_mask) -> Iterator[chess.Move]:
    if from_mask is None:
        from_mask = chess.BB_ALL
    if to_mask is None:
        to_mask = chess.BB_ALL
        
    return itertools.chain(
        generate_legal_moves(board,from_mask, to_mask & board.occupied_co[not board.turn]),
        generate_legal_ep(board,from_mask, to_mask))
        # board.generate_legal_moves(from_mask, to_mask & board.occupied_co[not board.turn]),
        # board.generate_legal_ep(from_mask, to_mask))

def _generate_evasions(object board, uint8_t king, uint64_t checkers, uint64_t from_mask, uint64_t to_mask) -> Iterator[chess.Move]:
        
    cdef uint8_t to_square
    cdef uint64_t sliders = checkers & (board.bishops | board.rooks | board.queens)
    cdef uint64_t attacked = 0
    cdef uint64_t target
    cdef uint8_t last_double
    cdef uint8_t checker
    cdef vector[uint8_t] attackedVec
    cdef vector[uint8_t] moveVec
    
    scan_reversed(sliders,attackedVec)
    cdef uint8_t size = attackedVec.size()
    
    #for checker in scan_reversed(sliders):
    for i in range(size):
        attacked |= ray(king, attackedVec[i]) & ~chess.BB_SQUARES[int(attackedVec[i])]

    if (1<<king) & from_mask:
        #if chess.BB_SQUARES[king] & from_mask:
        moveVec.clear()
        scan_reversed(chess.BB_KING_ATTACKS[king] & ~board.occupied_co[board.turn] & ~attacked & to_mask, moveVec)
        size = moveVec.size()
        for i in range(size):
            yield chess.Move(king, moveVec[i])
    
    # for checker in chess.scan_reversed(sliders):
    #     attacked |= chess.ray(king, checker) & ~chess.BB_SQUARES[checker]

    # if chess.BB_SQUARES[king] & from_mask:
    #     for to_square in chess.scan_reversed(chess.BB_KING_ATTACKS[king] & ~board.occupied_co[board.turn] & ~attacked & to_mask):
    #         yield chess.Move(king, to_square)

    checker = chess.msb(checkers)
    #checker = checkers.bit_length() - 1
    if chess.BB_SQUARES[checker] == checkers:
        # Capture or block a single checker.
        target = betweenPieces(king, checker) | checkers

        yield from generate_pseudo_legal_moves(board, ~board.kings & from_mask, target & to_mask)
        #yield from board.generate_pseudo_legal_moves(~board.kings & from_mask, target & to_mask)
        # Capture the checking pawn en passant (but avoid yielding
        # duplicate moves).
        if board.ep_square and not chess.BB_SQUARES[board.ep_square] & target:
            last_double = board.ep_square + (-8 if board.turn == chess.WHITE else 8)
            if last_double == checker:
                yield from board.generate_pseudo_legal_ep(from_mask, to_mask)

def generate_pseudo_legal_moves(object board, uint64_t from_mask, uint64_t to_mask) -> Iterator[chess.Move]:
    cdef uint64_t our_pieces = board.occupied_co[board.turn]
    cdef uint64_t all_pieces = board.occupied
    cdef uint8_t from_square
    cdef uint8_t to_square
    cdef uint64_t moves
    cdef uint64_t targets
    cdef uint64_t single_moves
    cdef uint64_t double_moves
    cdef vector[uint8_t] pieceVec
    cdef vector[uint8_t] pieceMoveVec
    cdef vector[uint8_t] pawnVec
    cdef vector[uint8_t] pawnMoveVec
    cdef vector[uint8_t] promotionVec

    cdef uint8_t outterSize
    cdef uint8_t innerSize
   
    # Generate piece moves.
    generatePieceMoves(pieceVec, pieceMoveVec, our_pieces, board.pawns, board.knights, board.bishops, board.rooks, board.queens, board.kings, board.occupied_co[True], board.occupied_co[False], board.occupied, from_mask, to_mask)
    outterSize = pieceVec.size()  
    #print(outterSize)
    for i in range(outterSize):
        yield chess.Move(pieceVec[i], pieceMoveVec[i])
    
    # Generate castling moves.
    if from_mask & board.kings:
        yield from board.generate_castling_moves(from_mask, to_mask)

    # The remaining moves are all pawn moves.
    cdef uint64_t pawns = board.pawns & board.occupied_co[board.turn] & from_mask
    if not pawns:
        return

    generatePawnMoves(pawnVec, pawnMoveVec, promotionVec, board.occupied_co[not board.turn], board.occupied, board.turn, pawns, from_mask, to_mask)
    outterSize = pawnVec.size()  
    for i in range(outterSize):
        if (promotionVec[i] == 1):
            yield chess.Move(pawnVec[i], pawnMoveVec[i])
        else:
            yield chess.Move(pawnVec[i], pawnMoveVec[i], promotionVec[i])

    # Generate en passant captures.
    if board.ep_square:
        yield from board.generate_pseudo_legal_ep(from_mask, to_mask)

def gives_check(object board,object move):
    board.push(move)
    try:
        return is_check(board.turn,board.occupied, board.queens | board.rooks, board.queens | board.bishops, board.kings, board.knights, board.pawns, board.occupied_co[not board.turn])    
    finally:
        board.pop()

          
def test1(board,square):
    #board.attacks_mask(square)
    #attackers_mask(board, not board.turn, square,board.occupied)
    #print(attackers_mask(board, not board.turn, square,board.occupied))
    #print(board.attacks_mask(square))
    #_slider_blockers(object board, uint8_t king)
    #print(board._slider_blockers(square))
    # cdef list captures = []
    # cdef object move
    # #cdef object moves = board.generate_legal_moves()
    # # Iterate through all legal moves
    
    # for move in generate_legal_captures(board,chess.BB_ALL,chess.BB_ALL):
    #     captures.append(move)
    # for move in generate_legal_moves(board,chess.BB_ALL,chess.BB_ALL):
    #     if move not in captures:
    #         pass
    # print(chess.ray(61,square))
    pass
    
    
def test2(object board, uint8_t square):
    #piece = board.piece_at(square)  
    #attacks_mask(bool(board.occupied_co[True] & (1<<square)),board.occupied,square,board.piece_type_at(square))
    #attackersMask(not board.turn, square, board.occupied, board.queens | board.rooks, board.queens | board.bishops, board.kings, board.knights, board.pawns, board.occupied_co[not board.turn])
    #print(attackersMask(not board.turn, square, board.occupied, board.queens | board.rooks, board.queens | board.bishops, board.kings, board.knights, board.pawns, board.occupied_co[not board.turn]))
    #print(attacks_mask(piece.color,board.occupied,square,piece.piece_type))
    #uint64_t slider_blockers(square, board.queens | board.rooks, board.queens | board.bishops, board.occupied_co[not board.turn], board.occupied_co[board.turn], board.occupied)
    #print(slider_blockers(square, board.queens | board.rooks, board.queens | board.bishops, board.occupied_co[not board.turn], board.occupied_co[board.turn], board.occupied))
    #is_capture(uint8_t from_square, uint8_t to_square, uint64_t occupied_co, bint is_en_passant)
    #cdef list captures = []
    # cdef object move
    # #cdef object moves = board.generate_legal_moves()
    # # Iterate through all legal moves
    
    # for move in generate_legal_captures(board,chess.BB_ALL,chess.BB_ALL):
    #     #captures.append(move)
    #     pass
    # for move in generate_legal_moves(board,chess.BB_ALL,chess.BB_ALL):
    #     if not is_capture(move.from_square, move.to_square, board.occupied_co[not board.turn], board.is_en_passant(move)):
    #         pass
    # print(ray(61,square))
    pass
    

def getRandomMove(board):
    legal_moves = list(board.legal_moves)
    #print(legal_moves)
    return random.choice(legal_moves) if legal_moves else None

def test3(int count):
    
    cdef int cur = 0
    
    cdef list chess_list = []
    cdef list my_list = []
    
    while (cur < count):
        board = chess.Board()
        while(not board.is_game_over()):
            chess_list = list(board.generate_legal_moves())
            my_list = list(generate_legal_moves(board,chess.BB_ALL,chess.BB_ALL))
            
            if (my_list != chess_list):
                print ("MY LIST: ", my_list)
                print()
                print ("THEIR LIST: ", chess_list)
                print(board)
                print(board.fen())
                return False
            
            board.push(getRandomMove(board))
        cur+=1
    return True

def test4 (board,increment):
    setAttackingLayer(board.occupied_co[True], board.occupied_co[False], board.kings, increment)
    printLayers()

# def test5(board,move):
#     isCapture = is_capture(move.from_square, move.to_square, board.occupied_co[not board.turn], board.is_en_passant(move))
#     zobrist = generateZobristHash(board.pawns, board.knights, board.bishops, board.rooks, board.queens, board.kings, board.occupied_co[True], board.occupied_co[False])
#     print(zobrist)
#     updateZobristHashForMove(zobrist, move.from_square, move.to_square, isCapture, board.pawns, board.knights, board.bishops, board.rooks, board.queens, board.kings, board.occupied_co[True], board.occupied_co[False])
#     print(zobrist)
# def test6(board,move):
#     zobrist = generateZobristHash(board.pawns, board.knights, board.bishops, board.rooks, board.queens, board.kings, board.occupied_co[True], board.occupied_co[False])
#     print(zobrist)
#     board.push(move)
#     zobrist = generateZobristHash(board.pawns, board.knights, board.bishops, board.rooks, board.queens, board.kings, board.occupied_co[True], board.occupied_co[False])
#     print(zobrist)
#     print(board)
    
def inititalize():
    initializeZobrist()
    initialize_attack_tables()