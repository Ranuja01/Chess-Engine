# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 12:50:22 2024

@author: Kumodth
"""

# main.py

import chess
import chess.pgn
import chess_eval  # Import the compiled Cython module
import ChessAI
import io
# Create a new chess board

pgn_string = """
1. e4 e5 2. Nf3 Nc6 3. Bb5 Nf6 4. O-O Nxe4 5. Re1 Nd6 6. Nxe5 Be7 7. Nxc6 dxc6 8. Ba4 O-O 9. c3 b5 10. Bc2 Bf5 11. d4 Bxc2 12. Qxc2 Nc4 13. Nd2 Nxd2 14. Qxd2 Bd6 15. Qd3 f5 16. Qf3 Qf6 17. Bf4 Bxf4 18. Qxf4 Rf7 19. Re2 g5 20. Qd2 Re7 21. Rae1 Rxe2 22. Qxe2 a5 23. Qh5 Qg6 24. Qxg6+ hxg6 25. Re6 Kh7 26. Rxc6 Rc8 27. b3 g4 28. g3 Kg7 29. Kg2 Kh7 30. f3 gxf3+ 31. Kxf3 b4 32. c4 g5 33. Rf6 c5 34. d5 Kg7 35. Rxf5 Kh6 36. Kg4 Rg8 37. Rxg5 Rxg5+ 38. Kf4 Rg8 39. Ke5 Re8+ 40. Kd6 Re2 41. Kxc5 Rxh2 42. d6 Rxa2 43. d7 Rd2 44. Kc6 a4 45. bxa4 Rd4 46. c5 b3 47. Kc7 b2 48. d8=Q Rxd8 49. Kxd8 b1=Q 50. c6 Qb8+ 51. Kd7 Qa8 52. c7 Qxa4+ 53. Kd8 Qd4+ 54. Kc8 Qd1 55. Kb8 Qb3+ 56. Kc8 Qxg3 57. Kb8 Qb3+ 58. Kc8 Qg8+ 59. Kb7 Qd5+ 60. Kb8 Qb5+ 61. Kc8
"""

# Create a PGN reader
pgn = chess.pgn.read_game(io.StringIO(pgn_string))

# Create a board from the game
board = pgn.board()

# Replay the moves to set the board to the final position
for move in pgn.mainline_moves():
    board.push(move)

board = chess.Board("r1q1r1k1/2pn1p1p/1p1bpnp1/p7/2PPPP2/2NB4/P1Q2P1P/R1B3RK b - - 0 15")

# Print the board in a human-readable format
print(board)


from ChessAI import ChessAI

from timeit import default_timer as timer
#from numba import njit
import easygui
import copy
import Rules
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, InputLayer
#from pickle import dump
from timeit import default_timer as timer
import chess
import chess.pgn
import io
import platform
import os
import chess_eval
import Cython_Chess

import sys
# Convert the board into a 12 channel tensor           
def encode_board(board):
    
    # Define piece mappings
    piece_to_channel = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
    }
    
    # Initialize a 12 channel tensor
    encoded_board = np.zeros((8, 8, 12), dtype=np.float32)
    
    # Populate the tensor
    for i in range(8):
        for j in range(8):
            # chess.square expects (file, rank) with 0-indexed file
            piece = board.piece_at(chess.square(j, 7-i))  
            if piece:
                channel = piece_to_channel[piece.symbol()]
                encoded_board[i, j, channel] = 1.0
    
    return encoded_board
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Set TensorFlow log level to suppress all but errors

pgnBoard = chess.Board()
pgnBoard.legal_moves

if platform.system() == 'Windows':
    data_path1 = '../Models/BlackModel4.keras'
    data_path2 = '../Models/WhiteModel1.keras'
    data_path3 = '../Models/WhiteEval_21_36.keras'
elif platform.system() == 'Linux':
    data_path1 = '/mnt/c/Users/Kumodth/Desktop/Programming/Chess Engine/Chess-Engine/Models/BlackModel4.keras'
    data_path2 = '/mnt/c/Users/Kumodth/Desktop/Programming/Chess Engine/Chess-Engine/Models/WhiteModel1.keras'
    data_path3 = '/mnt/c/Users/Kumodth/Desktop/Programming/Chess Engine/Chess-Engine/Models/WhiteEval_21_36.keras'

blackModel = tf.keras.models.load_model(data_path1)
whiteModel = tf.keras.models.load_model(data_path2)

model = tf.keras.models.load_model(data_path3)

# Assuming you have models already defined as black_model and white_model
chess_ai = ChessAI(blackModel, whiteModel, board)

#Cython_Chess.test4(board,5)

# board.push(chess.Move.from_uci("a2a3"))
# board.push(chess.Move.from_uci("e7e5"))
# board.push(chess.Move.from_uci("a3a4"))

# Call methods on the chess_ai instance

# Cython_Chess.inititalize()
# Cython_Chess.test4 (board,5)
# list1 = list(Cython_Chess.generate_pseudo_legal_moves2(board,chess.BB_ALL,chess.BB_ALL))
# list2 = list(Cython_Chess.generate_pseudo_legal_moves(board,chess.BB_ALL,chess.BB_ALL))
# list3 = list(board.generate_pseudo_legal_moves())
# print(len(list1), len(list2), len(list3))
# for i in range(len(list3)):
#     print (list1 [i], list2 [i], list3 [i])


# t0= timer()
# #for move in Cython_Chess.pseudo_legal_moves(board):
# for i in range (1):
#     for i in Cython_Chess.generate_pseudo_legal_moves2(board,chess.BB_ALL,chess.BB_ALL):
#         print(i)
#         # board.push(i)
#         # board.pop()
#         pass
# t1 = timer()
# print("Time elapsed: ", t1 - t0)

# t0= timer()
# for i in range (1):
#     for i in Cython_Chess.generate_pseudo_legal_moves(board,chess.BB_ALL,chess.BB_ALL):
#         print(i)
#         # board.push(i)
#         # board.pop()
#         pass
# t1 = timer()
# print("Time elapsed: ", t1 - t0)


# print("Eval: ", model.predict(np.array([encode_board(board)]),verbose=0))


# import Cython_Chess

# import cProfile
# import pstats
# t0= timer()
# def profile_alpha_beta():
#     result = chess_ai.alphaBetaWrapper(curDepth=0, depthLimit=5)

# profiler = cProfile.Profile()
# profiler.enable()
# profile_alpha_beta()
# profiler.disable()

# stats = pstats.Stats(profiler)
# stats.sort_stats(pstats.SortKey.TIME)
# stats.print_stats()
# t1 = timer()
# print("Time elapsed: ", t1 - t0)



# b2 = chess.Board("r1b2q1r/pp2kp2/2p1pNpp/8/2BQ1P2/bR2P3/P5PP/5RK1 b - - 7 18")
# print(chess_ai.ev(b2))

print(board, board.ply())
t0= timer()
result = chess_ai.alphaBetaWrapper(curDepth=0, depthLimit=6)
# print(result['a'],result['b'],result['c'],result['d'])
# print(f"Best score: {result['score']},")

t1 = timer()
print("Time elapsed: ", t1 - t0)
print(chess_ai.ev(board))
#a = chess.Board("r1q2rk1/pb1n1ppp/1ppbpn2/3P4/3P4/1PN1PN2/PB2BPPP/1R1Q1RK1 w - - 1 12")
# pgn_string = """
# 1. d4 { [%eval 0.17] } 1... d5 { [%eval 0.23] } 2. c4 { [%eval 0.19] } 2... e6 { [%eval 0.17] } 3. Nf3 { [%eval 0.16] } 3... Nf6 { [%eval 0.17] } 4. Nc3 { [%eval 0.1] } 4... c6 { [%eval 0.19] } 5. Bg5 { [%eval 0.14] } 5... h6 { [%eval 0.14] } 6. Bxf6 { [%eval 0.08] } 6... Qxf6 { [%eval 0.34] } 7. e3 { [%eval 0.11] } 7... Nd7 { [%eval 0.25] } 8. Bd3 { [%eval 0.13] } 8... dxc4 { [%eval 0.16] } 9. Bxc4 { [%eval 0.15] } 9... g6 { [%eval 0.15] } 10. O-O { [%eval 0.14] } 10... Bg7 { [%eval 0.23] } 11. Ne4 { [%eval 0.2] } 11... Qe7 { [%eval 0.15] } 12. Ne5 { [%eval -0.2] } 12... Nxe5 { [%eval -0.2] } 13. dxe5 { [%eval -0.23] } 13... Bxe5 { [%eval -0.21] } 14. f4 { [%eval -0.31] } 14... Bxb2 { [%eval -0.18] } 15. Rb1 { [%eval -0.18] } 15... Ba3?? { [%eval 2.3] } { Blunder. Bg7 was best. } (15... Bg7 16. Nd6+ Kf8 17. e4 b5 18. Bb3 c5 19. e5 a6 20. a4 Bd7 21. Qf3) 16. Qd4 { [%eval 2.04] } 16... Qf8?? { [%eval 4.76] } { Blunder. Rg8 was best. } (16... Rg8 17. Rb3) 17. Nf6+ { [%eval 4.19] } 17... Ke7 { [%eval 3.45] } 18. Rb3?! 
# """

# # Create a PGN reader
# pgn = chess.pgn.read_game(io.StringIO(pgn_string))

# # Create a board from the game
# a = pgn.board()
# for move in pgn.mainline_moves():
#     a.push(move)

# Cython_Chess.test4 (a,5)
# print(chess_ai.ev(a))
# print(a,a.ply())
# chess_ai = ChessAI(blackModel, whiteModel, a)
# t0= timer()
# result = chess_ai.alphaBetaWrapper(curDepth=0, depthLimit=5)
# print(result['a'],result['b'],result['c'],result['d'])
# print(f"Best score: {result['score']},")

# t1 = timer()
# print("Time elapsed: ", t1 - t0)

# t0= timer()
# print(Cython_Chess.test3(100))
# t1 = timer()
# print("Time elapsed: ", t1 - t0)




 


