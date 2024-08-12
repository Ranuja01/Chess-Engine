import time
import chess
import chess_eval
# Create a new board with a different FEN position

board = chess.Board()

# Define a target square (e.g., d5)\
    
board = chess.Board()
t0 = time.time()
target_square = chess.D5
chess_eval.find_legal_attackers(board, target_square)
t1 = time.time()

print("Time: ", t1 - t0)