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

board = chess.Board("3b1kQ1/P7/8/3p1p2/b7/P2R2P1/6K1/5n2 b - - 1 75")

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

def get_dict_size(d):
    total_size = sys.getsizeof(d)  # Size of the dictionary object itself
    for key, value in d.items():
        total_size += sys.getsizeof(key)  # Size of each key
        total_size += sys.getsizeof(value)  # Size of each value
    return total_size

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


# board.push(chess.Move.from_uci("a2a3"))
# board.push(chess.Move.from_uci("e7e5"))
# board.push(chess.Move.from_uci("a3a4"))

# Call methods on the chess_ai instance
Cython_Chess.inititalize()
t0= timer()
#for move in Cython_Chess.pseudo_legal_moves(board):
for i in range (1):
    for i in Cython_Chess.generate_legal_moves(board,chess.BB_ALL,chess.BB_ALL):
        print(i)
        pass
t1 = timer()
print("Time elapsed: ", t1 - t0)

t0= timer()
for i in range (1):
    for i in board.generate_legal_moves():
        print(i)
        pass
t1 = timer()
print("Time elapsed: ", t1 - t0)


# print("Eval: ", model.predict(np.array([encode_board(board)]),verbose=0))


# import Cython_Chess

# num_numbers = 1000000000
# num_threads = 160

# total_sum_single, single_thread_time, total_sum_multi, multi_thread_time = Cython_Chess.time_experiment(num_numbers, num_threads)

# print(f"Single-threaded sum: {total_sum_single:.6f}")
# print(f"Multi-threaded sum: {total_sum_multi:.6f}")
# print(f"Single-threaded time: {single_thread_time:.16f} seconds")
# print(f"Multi-threaded time: {multi_thread_time:.16f} seconds")



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

# t0= timer()
# result = chess_ai.alphaBetaWrapper(curDepth=0, depthLimit=4)
# print(result['a'],result['b'],result['c'],result['d'])
# print(f"Best score: {result['score']},")

# t1 = timer()
# print("Time elapsed: ", t1 - t0)

# t0= timer()
# result = chess_ai.alphaBetaWrapper(curDepth=0, depthLimit=5)
# print(result['a'],result['b'],result['c'],result['d'])
# print(f"Best score: {result['score']},")

# t1 = timer()
# print("Time elapsed: ", t1 - t0)

#a = chess.Board("r1q2rk1/pb1n1ppp/1ppbpn2/3P4/3P4/1PN1PN2/PB2BPPP/1R1Q1RK1 w - - 1 12")
# pgn_string = """
# 1. e4 e5 2. Nc3 Nf6 3. Nf3 Nc6 4. g3 Bc5 5. Bg2 d6 6. O-O a5 7. Qe1 O-O 8. d3 Nb4 9. Qd1 Bg4 10. a3 Nc6 11. h3 Bxf3 12. Bxf3 Qc8 13. Bg2 Bd4 14. Nb5 a4 15. c3 Bc5 16. d4 exd4 17. cxd4 Bb6 18. Bg5 Ra5 19. Qd3 Bxd4 20. Nxd4 Rxg5 21. f4 Rc5 22. Rad1 Na5 23. Rfe1 Nc4 24. Qc2 Ne3 25. Qxa4 Nxd1 26. Qxd1 Qe8 27. b4 Rc3 28. e5 dxe5 29. fxe5 Nd7 30. e6 fxe6 31. Nxe6 Rf7 32. Ng5 Re3 33. Ne4 Rxe1+ 34. Qxe1 c5 35. bxc5 h5 36. Qd1 Qe5 37. Nd6 Qxc5+ 38. Kh1 Re7 39. Ne4 Qxa3 40. Kh2 Qa2 41. Nc3 Qf2 42. Qd6 Rf7 43. Nd5 Kh7 44. Ne7 Nf8 45. Qe5 g6 46. h4 b5 47. Nxg6 Kxg6 48. Qg5+ Kh7 49. Qxh5+ Kg8 50. Qxb5 Rg7 51. Qb8 Rg4 52. Qe5 Kf7 53. Kh3 Rd4 54. Bd5+ Kg6 55. Be4+ Kf7 56. Bd5+ Kg6 57. Be4+ Kf7 58. Bd5+
# """

# # Create a PGN reader
# pgn = chess.pgn.read_game(io.StringIO(pgn_string))

# # Create a board from the game
# a = pgn.board()
# for move in pgn.mainline_moves():
#     a.push(move)
    
# chess_ai = ChessAI(blackModel, whiteModel, a)

# print(chess_ai.ev(a))
# print(a)
# result = chess_ai.alphaBetaWrapper(curDepth=0, depthLimit=5)
# print(result['a'],result['b'],result['c'],result['d'])

# print("2: ", a.is_repetition(2))
# print("3: ", a.is_repetition(3))
# print("3fold: ", a.can_claim_threefold_repetition())

# # Get bitboards for specific pieces
# white_pawns = board.pieces(chess.PAWN, chess.WHITE)
# black_pawns = board.pieces(chess.PAWN, chess.BLACK)
# white_rooks = board.pieces(chess.ROOK, chess.WHITE)
# black_rooks = board.pieces(chess.ROOK, chess.BLACK)
# bitboards = [occupied,white_pawns,black_pawns,white_rooks,black_rooks]
    
# # Call the Cython function
# #Cython_Chess.call_process_bitboards(bitboards)

# print(Cython_Chess.yield_msb(occupied))
    

# t0= timer()
# for i in chess.scan_forward(occupied):
#     #print(i)
#     pass
# t1 = timer()
# print("Time elapsed: ", t1 - t0)


# t0= timer()
# for i in range (1):
#     for i in chess.scan_reversed(board.occupied):
#         Cython_Chess.test1(board,i)
# t1 = timer()
# print("Time elapsed: ", t1 - t0)


# t0= timer()
# for i in range (1):
#     for i in chess.scan_reversed(board.occupied):
#         Cython_Chess.test2(board,i)
# t1 = timer()
# print("Time elapsed: ", t1 - t0)

t0= timer()
print(Cython_Chess.test3(100000))
t1 = timer()
print("Time elapsed: ", t1 - t0)
# t0= timer()
# for i in range(1000):
#     for move in board.generate_legal_moves():
#         if (board.is_capture(move)):
#             pass
#             #print(board.is_capture(move))
# t1 = timer()
# print("Time elapsed: ", t1 - t0)


# t0= timer()
# for i in range(1000):
#     for move in board.generate_legal_moves():
#         if (Cython_Chess.is_capture(board, move)):
#             pass
#             #print(Cython_Chess.is_capture(board,move))
# t1 = timer()
# print("Time elapsed: ", t1 - t0)

# # cache = chess_ai.get_move_cache()
# # size_in_bytes = get_dict_size(cache)
# # print(f"Size of dictionary in bytes: {size_in_bytes}")
# # print(f"Length of dictionary: {len(cache)}")


# # print(chess_ai.reorder_capture_moves())
# # print(chess_ai.get_legal_moves())
# # print(cache[board.occupied])

# t0= timer()
# result = chess_ai.alphaBetaWrapper(curDepth=0, depthLimit=6)
# print(result['a'],result['b'],result['c'],result['d'])
# print(f"Best score: {result['score']},")
# t1 = timer()
# print("Time elapsed: ", t1 - t0)

# t0= timer()
# result = chess_ai.alphaBetaWrapper(curDepth=0, depthLimit=7)
# print(result['a'],result['b'],result['c'],result['d'])
# print(f"Best score: {result['score']},")
# t1 = timer()
# print("Time elapsed: ", t1 - t0)

# t0= timer()
# result = chess_ai.alphaBetaWrapper(curDepth=0, depthLimit=7)
# print(result['a'],result['b'],result['c'],result['d'])
# print(f"Best score: {result['score']},")
# t1 = timer()
# print("Time elapsed: ", t1 - t0)

# cache = chess_ai.get_move_cache()
# size_in_bytes = get_dict_size(cache)
# print(f"Size of dictionary in bytes: {size_in_bytes}")
# print(f"Length of dictionary: {len(cache)}")

# t0= timer()
# result = chess_ai.alphaBetaWrapper(curDepth=0, depthLimit=5)
# print(result['a'],result['b'],result['c'],result['d'])
# print(f"Best score: {result['score']},")
# t1 = timer()
# print("Time elapsed: ", t1 - t0)

# t0= timer()
# result = chess_ai.alphaBetaWrapper(curDepth=0, depthLimit=6)
# print(result['a'],result['b'],result['c'],result['d'])
# print(f"Best score: {result['score']},")
# t1 = timer()
# print("Time elapsed: ", t1 - t0)

# # # Evaluate the board using Cython
# # score = chess_eval.evaluate_board(board)
# # print(f"Board Score: {score}")

# # Make a move (e.g., e2e4)

# board.push(chess.Move.from_uci("e7e5"))
# board.push(chess.Move.from_uci("f1c4"))

# t0= timer()
# result = chess_ai.alphaBetaWrapper(curDepth=0, depthLimit=5)
# print(result['a'],result['b'],result['c'],result['d'])
# print(f"Best score: {result['score']},")
# t1 = timer()
# print("Time elapsed: ", t1 - t0)


# board.push(chess.Move.from_uci("a7a6"))
# board.push(chess.Move.from_uci("d1f3"))

# t0= timer()
# result = chess_ai.alphaBetaWrapper(curDepth=0, depthLimit=6)
# print(result['a'],result['b'],result['c'],result['d'])
# print(f"Best score: {result['score']},")
# t1 = timer()
# print("Time elapsed: ", t1 - t0)

# board.push(chess.Move.from_uci("a6a5"))
# board.push(chess.Move.from_uci("f3f7"))
# # Print the board after the move
# print(board)

# # Evaluate the new board state
# score = chess_eval.evaluate_board(board)
# print(f"New Board Score: {score}")
