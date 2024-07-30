# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 12:50:22 2024

@author: Kumodth
"""

# main.py

import chess
import chess_eval  # Import the compiled Cython module

# Create a new chess board
board = chess.Board()

# Print the board in a human-readable format
print(board)

# Evaluate the board using Cython
score = chess_eval.evaluate_board(board)
print(f"Board Score: {score}")

# Make a move (e.g., e2e4)
board.push(chess.Move.from_uci("e2e4"))
board.push(chess.Move.from_uci("e7e5"))
board.push(chess.Move.from_uci("f1c4"))
board.push(chess.Move.from_uci("a7a6"))
board.push(chess.Move.from_uci("d1f3"))
board.push(chess.Move.from_uci("a6a5"))
board.push(chess.Move.from_uci("f3f7"))
# Print the board after the move
print(board)

# Evaluate the new board state
score = chess_eval.evaluate_board(board)
print(f"New Board Score: {score}")
