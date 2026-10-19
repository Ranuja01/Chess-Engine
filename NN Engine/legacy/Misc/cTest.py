# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 01:13:48 2024

@author: Kumodth
"""

import chess


Color: TypeAlias = bool
WHITE: Color = True
BLACK: Color = False
COLORS: List[Color] = [WHITE, BLACK]
COLOR_NAMES: List[ColorName] = ["black", "white"]

PieceType: TypeAlias = int
PAWN: PieceType = 1
KNIGHT: PieceType = 2
BISHOP: PieceType = 3
ROOK: PieceType = 4
QUEEN: PieceType = 5
KING: PieceType = 6
PIECE_TYPES: List[PieceType] = [PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING]
PIECE_SYMBOLS = [None, "p", "n", "b", "r", "q", "k"]
PIECE_NAMES = [None, "pawn", "knight", "bishop", "rook", "queen", "king"]


UNICODE_PIECE_SYMBOLS = {
    "R": "♖", "r": "♜",
    "N": "♘", "n": "♞",
    "B": "♗", "b": "♝",
    "Q": "♕", "q": "♛",
    "K": "♔", "k": "♚",
    "P": "♙", "p": "♟",
}

FILE_NAMES = ["a", "b", "c", "d", "e", "f", "g", "h"]

RANK_NAMES = ["1", "2", "3", "4", "5", "6", "7", "8"]

STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
"""The FEN for the standard chess starting position."""

STARTING_BOARD_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
"""The board part of the FEN for the standard chess starting position."""

class InvalidMoveError(ValueError):
    """Raised when move notation is not syntactically valid"""


class IllegalMoveError(ValueError):
    """Raised when the attempted move is illegal in the current position"""


class AmbiguousMoveError(ValueError):
    """Raised when the attempted move is ambiguous in the current position"""


Square: TypeAlias = int
A1: Square = 0
B1: Square = 1
C1: Square = 2
D1: Square = 3
E1: Square = 4
F1: Square = 5
G1: Square = 6
H1: Square = 7
A2: Square = 8
B2: Square = 9
C2: Square = 10
D2: Square = 11
E2: Square = 12
F2: Square = 13
G2: Square = 14
H2: Square = 15
A3: Square = 16
B3: Square = 17
C3: Square = 18
D3: Square = 19
E3: Square = 20
F3: Square = 21
G3: Square = 22
H3: Square = 23
A4: Square = 24
B4: Square = 25
C4: Square = 26
D4: Square = 27
E4: Square = 28
F4: Square = 29
G4: Square = 30
H4: Square = 31
A5: Square = 32
B5: Square = 33
C5: Square = 34
D5: Square = 35
E5: Square = 36
F5: Square = 37
G5: Square = 38
H5: Square = 39
A6: Square = 40
B6: Square = 41
C6: Square = 42
D6: Square = 43
E6: Square = 44
F6: Square = 45
G6: Square = 46
H6: Square = 47
A7: Square = 48
B7: Square = 49
C7: Square = 50
D7: Square = 51
E7: Square = 52
F7: Square = 53
G7: Square = 54
H7: Square = 55
A8: Square = 56
B8: Square = 57
C8: Square = 58
D8: Square = 59
E8: Square = 60
F8: Square = 61
G8: Square = 62
H8: Square = 63
SQUARES: List[Square] = list(range(64))

SQUARE_NAMES = [f + r for r in RANK_NAMES for f in FILE_NAMES]

def parse_square(name: str) -> Square:
    """
    Gets the square index for the given square *name*
    (e.g., ``a1`` returns ``0``).

    :raises: :exc:`ValueError` if the square name is invalid.
    """
    return SQUARE_NAMES.index(name)

def square_name(square: Square) -> str:
    """Gets the name of the square, like ``a3``."""
    return SQUARE_NAMES[square]

def square(file_index: int, rank_index: int) -> Square:
    """Gets a square number by file and rank index."""
    return rank_index * 8 + file_index

def square_file(square: Square) -> int:
    """Gets the file index of the square where ``0`` is the a-file."""
    return square & 7

def square_rank(square: Square) -> int:
    """Gets the rank index of the square where ``0`` is the first rank."""
    return square >> 3

def square_distance(a: Square, b: Square) -> int:
    """
    Gets the Chebyshev distance (i.e., the number of king steps) from square *a* to *b*.
    """
    return max(abs(square_file(a) - square_file(b)), abs(square_rank(a) - square_rank(b)))

def square_manhattan_distance(a: Square, b: Square) -> int:
    """
    Gets the Manhattan/Taxicab distance (i.e., the number of orthogonal king steps) from square *a* to *b*.
    """
    return abs(square_file(a) - square_file(b)) + abs(square_rank(a) - square_rank(b))

def square_mirror(square: Square) -> Square:
    """Mirrors the square vertically."""
    return square ^ 0x38

SQUARES_180: List[Square] = [square_mirror(sq) for sq in SQUARES]


Bitboard: TypeAlias = int
BB_EMPTY: Bitboard = 0
BB_ALL: Bitboard = 0xffff_ffff_ffff_ffff

BB_A1: Bitboard = 1 << A1
BB_B1: Bitboard = 1 << B1
BB_C1: Bitboard = 1 << C1
BB_D1: Bitboard = 1 << D1
BB_E1: Bitboard = 1 << E1
BB_F1: Bitboard = 1 << F1
BB_G1: Bitboard = 1 << G1
BB_H1: Bitboard = 1 << H1
BB_A2: Bitboard = 1 << A2
BB_B2: Bitboard = 1 << B2
BB_C2: Bitboard = 1 << C2
BB_D2: Bitboard = 1 << D2
BB_E2: Bitboard = 1 << E2
BB_F2: Bitboard = 1 << F2
BB_G2: Bitboard = 1 << G2
BB_H2: Bitboard = 1 << H2
BB_A3: Bitboard = 1 << A3
BB_B3: Bitboard = 1 << B3
BB_C3: Bitboard = 1 << C3
BB_D3: Bitboard = 1 << D3
BB_E3: Bitboard = 1 << E3
BB_F3: Bitboard = 1 << F3
BB_G3: Bitboard = 1 << G3
BB_H3: Bitboard = 1 << H3
BB_A4: Bitboard = 1 << A4
BB_B4: Bitboard = 1 << B4
BB_C4: Bitboard = 1 << C4
BB_D4: Bitboard = 1 << D4
BB_E4: Bitboard = 1 << E4
BB_F4: Bitboard = 1 << F4
BB_G4: Bitboard = 1 << G4
BB_H4: Bitboard = 1 << H4
BB_A5: Bitboard = 1 << A5
BB_B5: Bitboard = 1 << B5
BB_C5: Bitboard = 1 << C5
BB_D5: Bitboard = 1 << D5
BB_E5: Bitboard = 1 << E5
BB_F5: Bitboard = 1 << F5
BB_G5: Bitboard = 1 << G5
BB_H5: Bitboard = 1 << H5
BB_A6: Bitboard = 1 << A6
BB_B6: Bitboard = 1 << B6
BB_C6: Bitboard = 1 << C6
BB_D6: Bitboard = 1 << D6
BB_E6: Bitboard = 1 << E6
BB_F6: Bitboard = 1 << F6
BB_G6: Bitboard = 1 << G6
BB_H6: Bitboard = 1 << H6
BB_A7: Bitboard = 1 << A7
BB_B7: Bitboard = 1 << B7
BB_C7: Bitboard = 1 << C7
BB_D7: Bitboard = 1 << D7
BB_E7: Bitboard = 1 << E7
BB_F7: Bitboard = 1 << F7
BB_G7: Bitboard = 1 << G7
BB_H7: Bitboard = 1 << H7
BB_A8: Bitboard = 1 << A8
BB_B8: Bitboard = 1 << B8
BB_C8: Bitboard = 1 << C8
BB_D8: Bitboard = 1 << D8
BB_E8: Bitboard = 1 << E8
BB_F8: Bitboard = 1 << F8
BB_G8: Bitboard = 1 << G8
BB_H8: Bitboard = 1 << H8
BB_SQUARES: List[Bitboard] = [1 << sq for sq in SQUARES]

BB_CORNERS: Bitboard = BB_A1 | BB_H1 | BB_A8 | BB_H8
BB_CENTER: Bitboard = BB_D4 | BB_E4 | BB_D5 | BB_E5

BB_LIGHT_SQUARES: Bitboard = 0x55aa_55aa_55aa_55aa
BB_DARK_SQUARES: Bitboard = 0xaa55_aa55_aa55_aa55

BB_FILE_A: Bitboard = 0x0101_0101_0101_0101 << 0
BB_FILE_B: Bitboard = 0x0101_0101_0101_0101 << 1
BB_FILE_C: Bitboard = 0x0101_0101_0101_0101 << 2
BB_FILE_D: Bitboard = 0x0101_0101_0101_0101 << 3
BB_FILE_E: Bitboard = 0x0101_0101_0101_0101 << 4
BB_FILE_F: Bitboard = 0x0101_0101_0101_0101 << 5
BB_FILE_G: Bitboard = 0x0101_0101_0101_0101 << 6
BB_FILE_H: Bitboard = 0x0101_0101_0101_0101 << 7
BB_FILES: List[Bitboard] = [BB_FILE_A, BB_FILE_B, BB_FILE_C, BB_FILE_D, BB_FILE_E, BB_FILE_F, BB_FILE_G, BB_FILE_H]

BB_RANK_1: Bitboard = 0xff << (8 * 0)
BB_RANK_2: Bitboard = 0xff << (8 * 1)
BB_RANK_3: Bitboard = 0xff << (8 * 2)
BB_RANK_4: Bitboard = 0xff << (8 * 3)
BB_RANK_5: Bitboard = 0xff << (8 * 4)
BB_RANK_6: Bitboard = 0xff << (8 * 5)
BB_RANK_7: Bitboard = 0xff << (8 * 6)
BB_RANK_8: Bitboard = 0xff << (8 * 7)
BB_RANKS: List[Bitboard] = [BB_RANK_1, BB_RANK_2, BB_RANK_3, BB_RANK_4, BB_RANK_5, BB_RANK_6, BB_RANK_7, BB_RANK_8]

BB_BACKRANKS: Bitboard = BB_RANK_1 | BB_RANK_8



def pseudo_legal_moves(object board):
        """
        A dynamic list of pseudo-legal moves, much like the legal move list.

        Pseudo-legal moves might leave or put the king in check, but are
        otherwise valid. Null moves are not pseudo-legal. Castling moves are
        only included if they are completely legal.

        Wraps :func:`~chess.Board.generate_pseudo_legal_moves()` and
        :func:`~chess.Board.is_pseudo_legal()`.
        """
        return generate_pseudo_legal_moves(board)


cdef list generate_pseudo_legal_moves(object board):
    
    cdef list moves_list = []
    
    cdef unsigned long long our_pieces = board.occupied_co[board.turn]

    # Generate piece moves.
    cdef unsigned long long non_pawns = our_pieces & ~board.pawns & BB_ALL
    for from_square in scan_reversed(non_pawns):
        moves = board.attacks_mask(from_square) & ~our_pieces & BB_ALL
        for to_square in scan_reversed(moves):
            moves_list.append(chess.Move(from_square, to_square))

    # Generate castling moves.
    if BB_ALL & board.kings:
        moves_list.extend(board.generate_castling_moves(BB_ALL, BB_ALL))

    # The remaining moves are all pawn moves.
    cdef unsigned long long pawns = board.pawns & board.occupied_co[board.turn] & BB_ALL
    if not pawns:
        return

    # Generate pawn captures.
    cdef unsigned long long capturers = pawns
    for from_square in scan_reversed(capturers):
        targets = (
            chess.BB_PAWN_ATTACKS[board.turn][from_square] &
            board.occupied_co[not board.turn] & BB_ALL)

        for to_square in scan_reversed(targets):
            if chess.square_rank(to_square) in [0, 7]:
                moves_list.append( chess.Move(from_square, to_square, 5))
                moves_list.append( chess.Move(from_square, to_square, 4))
                moves_list.append( chess.Move(from_square, to_square, 3))
                moves_list.append( chess.Move(from_square, to_square, 2))
            else:
                moves_list.append( chess.Move(from_square, to_square))

    # Prepare pawn advance generation.
    if board.turn:
        single_moves = pawns << 8 & ~board.occupied
        double_moves = single_moves << 8 & ~board.occupied & (BB_RANK_3 | BB_RANK_4)
    else:
        single_moves = pawns >> 8 & ~board.occupied
        double_moves = single_moves >> 8 & ~board.occupied & (BB_RANK_6 | BB_RANK_5)

    single_moves &= BB_ALL
    double_moves &= BB_ALL

    # Generate single pawn moves.
    for to_square in scan_reversed(single_moves):
        from_square = to_square + (8 if not board.turn else -8)

        if chess.square_rank(to_square) in [0, 7]:
            moves_list.append( chess.Move(from_square, to_square, 5))
            moves_list.append( chess.Move(from_square, to_square, 4))
            moves_list.append( chess.Move(from_square, to_square, 3))
            moves_list.append( chess.Move(from_square, to_square, 2))
        else:
            moves_list.append( chess.Move(from_square, to_square))

    # Generate double pawn moves.
    for to_square in scan_reversed(double_moves):
        from_square = to_square + (16 if not board.turn else -16)
        moves_list.append( chess.Move(from_square, to_square))

    # Generate en passant captures.
    if board.ep_square:
        moves_list.extend( board.generate_pseudo_legal_ep(BB_ALL, BB_ALL))
        
    return moves_list

cdef list generate_pseudo_legal_ep(object board):
    
    cdef list moves_list = []
    if not board.ep_square or not BB_SQUARES[board.ep_square] & BB_ALL:
        return

    if BB_SQUARES[board.ep_square] & board.occupied:
        return

    capturers = (
        board.pawns & board.occupied_co[board.turn] & BB_ALL &
        chess.BB_PAWN_ATTACKS[not board.turn][board.ep_square] &
        BB_RANKS[4 if board.turn else 3])

    for capturer in scan_reversed(capturers):
        moves_list.append( chess.Move(capturer, board.ep_square))
    return moves_list
        
def scan_reversed(object bb) -> Iterator[Square]:
    while bb:
        r = bb.bit_length() - 1
        yield r
        bb ^= BB_SQUARES[r]