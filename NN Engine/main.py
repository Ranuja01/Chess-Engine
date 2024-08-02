# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 12:50:22 2024

@author: Kumodth
"""

# main.py

import chess
import chess_eval  # Import the compiled Cython module
import ChessAI

# Create a new chess board
board = chess.Board()

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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Set TensorFlow log level to suppress all but errors

pgnBoard = chess.Board()
pgnBoard.legal_moves
if platform.system() == 'Windows':
    data_path1 = '../Models/BlackModel4.keras'
    data_path2 = '../Models/WhiteModel1.keras'
elif platform.system() == 'Linux':
    data_path1 = '/mnt/c/Users/Kumodth/Desktop/Programming/Chess Engine/Chess-Engine/Models/BlackModel4.keras'
    data_path2 = '/mnt/c/Users/Kumodth/Desktop/Programming/Chess Engine/Chess-Engine/Models/WhiteModel1.keras'
    

blackModel = tf.keras.models.load_model(data_path1)
whiteModel = tf.keras.models.load_model(data_path2)

# Assuming you have models already defined as black_model and white_model
chess_ai = ChessAI(blackModel, whiteModel, board)


board.push(chess.Move.from_uci("e2e4"))
# Call methods on the chess_ai instance



import cProfile
import pstats
t0= timer()
def profile_alpha_beta():
    chess_ai.alphaBetaWrapper(curDepth=0, depthLimit=5)

profiler = cProfile.Profile()
profiler.enable()
profile_alpha_beta()
profiler.disable()

stats = pstats.Stats(profiler)
stats.sort_stats(pstats.SortKey.TIME)
stats.print_stats()

t1 = timer()
print("Time elapsed: ", t1 - t0)

t0= timer()
result = chess_ai.alphaBetaWrapper(curDepth=0, depthLimit=4)
print(result['a'],result['b'],result['c'],result['d'])
print(f"Best score: {result['score']},")
t1 = timer()
print("Time elapsed: ", t1 - t0)

# Evaluate the board using Cython
score = chess_eval.evaluate_board(board)
print(f"Board Score: {score}")

# Make a move (e.g., e2e4)

board.push(chess.Move.from_uci("e7e5"))
board.push(chess.Move.from_uci("f1c4"))

t0= timer()
result = chess_ai.alphaBetaWrapper(curDepth=0, depthLimit=5)
print(result['a'],result['b'],result['c'],result['d'])
print(f"Best score: {result['score']},")
t1 = timer()
print("Time elapsed: ", t1 - t0)


board.push(chess.Move.from_uci("a7a6"))
board.push(chess.Move.from_uci("d1f3"))

t0= timer()
result = chess_ai.alphaBetaWrapper(curDepth=0, depthLimit=6)
print(result['a'],result['b'],result['c'],result['d'])
print(f"Best score: {result['score']},")
t1 = timer()
print("Time elapsed: ", t1 - t0)

board.push(chess.Move.from_uci("a6a5"))
board.push(chess.Move.from_uci("f3f7"))
# Print the board after the move
print(board)

# Evaluate the new board state
score = chess_eval.evaluate_board(board)
print(f"New Board Score: {score}")
