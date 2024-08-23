# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 15:10:14 2024

@author: Kumodth
"""

import tensorflow as tf
import numpy as np
import chess
import chess.engine
import chess.pgn
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, InputLayer, Dropout, BatchNormalization, GlobalAveragePooling2D, MaxPooling2D, MultiHeadAttention, Add, LayerNormalization
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from tensorflow.keras.metrics import TopKCategoricalAccuracy
from tensorflow.keras.optimizers import Adam, Nadam, RMSprop
from timeit import default_timer as timer
from tensorflow.keras.losses import KLDivergence
from copy import deepcopy
import io
import platform
import gc

from numba import cuda
from numba import njit
import itertools
import threading
import time
import copy
import os

tf.config.optimizer.set_jit(True)  # Enable XLA

trainingCount = 0
loop = True
lock = threading.Lock()
event = threading.Event()
boards = []
if platform.system() == 'Windows':
    data_path1 = r'../Models/BlackModel_21_36(11)_RL(3)_selfplay_SGD.keras'
    data_path2 = r'../Models/WhiteModel_21_36(11)_RL(3)_selfplay_SGD.keras'

elif platform.system() == 'Linux':
    
    data_path1 = r'/mnt/c/Users/Kumodth/Desktop/Programming/Chess Engine/Chess-Engine/Models/BlackModel_21_36(11)_RL(3)_selfplay_SGD.keras'
    data_path2 = r'/mnt/c/Users/Kumodth/Desktop/Programming/Chess Engine/Chess-Engine/Models/WhiteModel_21_36(11)_RL(3)_selfplay_SGD.keras'
    
blackModel = tf.keras.models.load_model(data_path1)
whiteModel = tf.keras.models.load_model(data_path2)

# Set the path to the Stockfish binary
if platform.system() == 'Windows':
    STOCKFISH_PATH = "../../stockfish/stockfish-windows-x86-64-avx2"  # Make sure this points to your Stockfish binary
elif platform.system() == 'Linux':
    STOCKFISH_PATH = "/mnt/c/Users/Kumodth/Desktop/Programming/Chess Engine/stockfish/stockfish-windows-x86-64-avx2.exe"  # Make sure this points to your Stockfish binary

def is_promotion_move_enhanced(move, board):
    """
    Checks if a move is a promotion move.

    Parameters:
    - move: chess.Move object to be checked
    - board: chess.Board object for the current state

    Returns:
    - True if the move is a promotion, False otherwise
    """
    if move.promotion is not None:
        return True
    
    # Check if a pawn is moving to the last rank
    from_square = move.from_square
    to_square = move.to_square
    piece = board.piece_at(from_square)

    if piece and piece.piece_type == chess.PAWN:
        # Check if the move is to the promotion rank
        if (piece.color == chess.WHITE and chess.square_rank(to_square) == 7) or \
           (piece.color == chess.BLACK and chess.square_rank(to_square) == 0):
            return True

    return False

def suggest_moves(board, engine, time_limit=0.0001, depth=None, multipv=1):
    """
    Suggests the best move(s) for the given position.

    Parameters:
    - board: chess.Board object representing the current position.
    - engine: chess.engine.SimpleEngine instance connected to Stockfish.
    - time_limit: float, maximum time in seconds to analyze each position.
    - depth: int, optional depth limit for the analysis.
    - multipv: int, number of top moves to suggest (1 for the best move, >1 for more).

    Returns:
    - suggestions: list of tuples (move, score), sorted by best move.
    """
    # Set analysis parameters
    limit = chess.engine.Limit(time=time_limit, depth=depth)
    result = engine.analyse(board, limit, multipv=multipv)

    # Parse results to extract moves and evaluations
    suggestions = []
    for entry in result:
        move = entry["pv"][0]  # Principal variation's first move
        #score = entry["score"].white().score(mate_score=10000) / 100.0
        suggestions.append(move)

    # Sort suggestions by evaluation score (higher is better for White)
    #suggestions.sort(key=lambda x: x[1], reverse=True)

    return suggestions

def get_stockfish_evaluation(board, engine, time_limit=0.01):
    """
    Get Stockfish evaluation of the given board position.

    Parameters:
    - board: chess.Board object
    - engine: chess.engine.SimpleEngine instance
    - time_limit: float, time to analyze the position in seconds

    Returns:
    - evaluation: float, centipawn evaluation or large numerical value for mate
    """
    result = engine.analyse(board, chess.engine.Limit(time=time_limit, depth = 20))
    score = result["score"].relative

    # Convert Mate in X to a large numerical value
    if score.is_mate():
        mate_in_moves = score.mate()
        evaluation = 100 * (1 - mate_in_moves/200) if mate_in_moves > 0 else -100 * (1 + mate_in_moves/200)
    else:
        evaluation = score.score(mate_score=10000) / 100.0
    
    return evaluation

def getNNMove(board):
    filteredPrediction = [0]*4096
    inputBoard = [encode_board(board)]
    if board.turn:
        
        prediction = whiteModel.predict(np.array(inputBoard),verbose=0)
        
        # Filter the predictions to only contain legal moves
        for move in board.legal_moves:
            cur = str(move)
            a = ord(cur[0]) - 96
            b = int(cur[1])
            c = ord(cur[2]) - 96
            d = int(cur[3])                       
            filteredPrediction[reversePrediction(a,b,c,d) - 1] = prediction[0][reversePrediction(a,b,c,d) - 1]
                         
        filteredPrediction = np.array(filteredPrediction)
        
        a,b,c,d = predictionInfo(np.argmax(filteredPrediction))
        
    else:
    
        prediction = blackModel.predict(np.array(inputBoard), verbose=0)
        
        # Filter the predictions to only contain legal moves
        for move in board.legal_moves:
            cur = str(move)
            a = ord(cur[0]) - 96
            b = int(cur[1])
            c = ord(cur[2]) - 96
            d = int(cur[3])                       
            filteredPrediction[reversePrediction(a,b,c,d) - 1] = prediction[0][reversePrediction(a,b,c,d) - 1]
                         
        filteredPrediction = np.array(filteredPrediction)
        
        a,b,c,d = predictionInfo(np.argmax(filteredPrediction))

    a = chr(a + 96)
    b = str(b)
    c = chr(c + 96)
    d = str(d)
    
    if ((a,b,c,d) == ('a','1','a','1')):
        return getRandomMove(board)
    
    if (is_promotion_move_enhanced(move.from_uci(a+b+c+d),board)):
        return move.from_uci(a+b+c+d+'q')
    else:
        return move.from_uci(a+b+c+d)

def getRandomMove(board):
    legal_moves = list(board.legal_moves)
    #print(legal_moves)
    return random.choice(legal_moves) if legal_moves else None

# Function to convert the neural network output to 4 coordinates
def predictionInfo(prediction):
    
    # Get the starting square by integer dividing by 64
    # This is because the encoding uses multiples of 64 to represent each starting square going to each other
    pieceToBeMoved = prediction // 64
    
    # Get location square via the remainder, following the same logic as above
    squareToBeMovedTo = prediction % 64

    # Acquire the row and coloumns by utilizing the same technique as above
    pieceToBeMovedXLocation = pieceToBeMoved // 8 + 1
    pieceToBeMovedYLocation = pieceToBeMoved % 8 + 1
    
    # Coordinates of the square to be moved to
    squareToBeMovedToXLocation = squareToBeMovedTo // 8 + 1
    squareToBeMovedToYLocation = squareToBeMovedTo % 8 + 1
    
    return pieceToBeMovedXLocation, pieceToBeMovedYLocation, squareToBeMovedToXLocation, squareToBeMovedToYLocation

# Turns the coordinates back into the NN output
def reversePrediction(x,y,i,j):
    # First acquire the starting square number and multiply by 64 to get its base number
    # Then add the remaining starting point of the location to be moved to
    return (((x - 1) * 8 + y) - 1)  *64 + ((i - 1) * 8 + j)
           
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

def reflect_board(board):
    # Create a new board which is a reflection of the input board
    reflected_board = chess.Board()
    reflected_board.clear()  # Clear the board first
    
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            # Calculate the reflected square
            file = chess.square_file(square)
            rank = chess.square_rank(square)
            reflected_file = 7 - file
            reflected_square = chess.square(reflected_file, rank)
            reflected_board.set_piece_at(reflected_square, piece)
    
    return reflected_board

# Function to compute the policy gradient and update the model
def update_policy(policy_model, states, actions, rewards):
    #print(states)
    print(actions)
    # Define the optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.5)
    
    with tf.GradientTape(watch_accessed_variables=True, persistent=True) as tape:
        # Compute the loss for the trajectory
        
        action_probs = policy_model(states, training=True)
        print(action_probs)
        action_probs = tf.clip_by_value(action_probs, 1e-7, 1.0)  # Avoid log(0)
        action_indices = tf.stack([tf.range(len(actions)), actions], axis=-1)
        log_probs = tf.math.log(tf.gather_nd(action_probs, action_indices))
        loss = -tf.reduce_mean(log_probs * rewards)  # Negative because we maximize reward
        tape.watch(loss)
        #tape.watch(policy_model.trainable_variables)
            
        print(loss)
        # Compute and apply gradients
    grads = tape.gradient(loss, policy_model.trainable_variables)
    for grad, var in zip(grads, policy_model.trainable_variables):
        print(f'Gradient for {var.name}: {tf.reduce_mean(grad)}')
    optimizer.apply_gradients(zip(grads, policy_model.trainable_variables))

def take_action(board, engine, move, prev_turn_eval):
    
    prev_eval = get_stockfish_evaluation(board, engine, 0.005)
    board.push(move)
    cur_eval = get_stockfish_evaluation(board, engine, 0.005) * -1
    #print(prev_eval,cur_eval)
    if (prev_eval - cur_eval > 0.1):
        return (cur_eval - prev_eval), cur_eval
    else:
        return (cur_eval - prev_turn_eval), cur_eval

def compute_discounted_rewards(rewards, gamma):
    discounted_rewards = np.zeros_like(rewards, dtype=np.float32)
    cumulative_reward = 0.0
    for t in reversed(range(len(rewards))):
        cumulative_reward = rewards[t] + gamma * cumulative_reward
        discounted_rewards[t] = cumulative_reward
    return discounted_rewards
        

def selfPlay():
     
    stockfish_path=STOCKFISH_PATH
    global boards
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    #engine.configure({"Threads": 4, "Hash": 4096})
    for i in range (1):
        board = chess.Board()
        black_states, black_actions, black_rewards = [], [], []
        white_states, white_actions, white_rewards = [], [], []
        prev_black_eval = 0.0
        prev_white_eval = 0.0
        inGameCount = 0
        while(not(board.is_game_over())):
        
            # Generate a random number between 0 and 1
            random_number = random.random()
            
            stockfish_usage = 0.85
            
            # Define the probability thresholds for each branch
            if random_number < stockfish_usage:  # 50% chance for the first branch                                
                move = suggest_moves(board, engine, time_limit=0.001, depth=15, multipv=1)[0]                
            elif random_number < 0.95:  # 45% chance for the second branch (0.2 + 0.3)                
                move = getNNMove(board)
            else:  # 5% chance for the third branch (0.5 + 0.5)
                move = getRandomMove(board)
            
            if (board.turn):
                reward, prev_white_eval = take_action(board, engine, move, prev_white_eval)
                
                if (inGameCount >= 20 and inGameCount <= 36):
                    # Store the trajectory
                    move = board.pop()
                    white_states.append(encode_board(board))
                    board.push(move)
                    moveMade = str(board.peek())
                    a = ord(moveMade[0:1]) - 96
                    b = int(moveMade[1:2])
                    c = ord(moveMade[2:3]) - 96
                    d = int(moveMade[3:4])
                    
                    white_actions.append(reversePrediction(a,b,c,d) - 1)
                    white_rewards.append(reward)
            else:
                reward, prev_black_eval = take_action(board, engine, move, prev_black_eval)

                if (inGameCount >= 20 and inGameCount <= 36):
                    # Store the trajectory
                    move = board.pop()
                    black_states.append(encode_board(board))
                    board.push(move)
                    moveMade = str(board.peek())
                    a = ord(moveMade[0:1]) - 96
                    b = int(moveMade[1:2])
                    c = ord(moveMade[2:3]) - 96
                    d = int(moveMade[3:4])
                    
                    black_actions.append(reversePrediction(a,b,c,d) - 1)
                    black_rewards.append(reward)                
                    boards.append(copy.deepcopy(board))
            inGameCount += 1
        
        # Convert lists to tensors
        black_states = np.array(black_states)
        black_actions = np.array(black_actions)
        black_rewards = np.array(black_rewards)
        
        white_states = np.array(white_states)
        white_actions = np.array(white_actions)
        white_rewards = np.array(white_rewards)
        
        # Compute returns (discounted cumulative rewards)
        black_discounted_rewards = compute_discounted_rewards(black_rewards, gamma=0.99)
        white_discounted_rewards = compute_discounted_rewards(white_rewards, gamma=0.99)
        
        # Update the policy using the observed trajectory
        if (len(black_rewards) > 0 and len(white_rewards) > 0):
            with lock:
                update_policy(blackModel, black_states, black_actions, black_discounted_rewards)
            with lock:
                update_policy(whiteModel, white_states, white_actions, white_discounted_rewards)

        if (i + 1 % 20) == 0:
            del black_states, black_actions, black_rewards, white_states, white_actions, white_rewards
            gc.collect()        

    engine.close()

# List to hold the thread objects
threads = []

# Create and start threads
for i in range(1):  # Example with 5 threads
    t = threading.Thread(target=selfPlay, args=())
    t.start()
    threads.append(t)

# Main function continues to execute
print("Main function is doing other things...")

# Wait for threads to complete
count = 0

t0_full = timer()
t0 = timer()

# Wait for all threads to complete
for t in threads:
    t.join()
    
print("All threads have finished.")
t1_full = timer()
print("Time elapsed: ", t1_full - t0_full)
    
if platform.system() == 'Windows':
    data_path = r'../Models/WhiteModel6_MidEndGame(8)_Refined.keras'
elif platform.system() == 'Linux':
    data_path1 = '/mnt/c/Users/Kumodth/Desktop/Programming/Chess Engine/Chess-Engine/Models/WhiteModel_21_36(11)_RL(3)_selfplay_SGD.keras'  # Example for WSL
    data_path2 = '/mnt/c/Users/Kumodth/Desktop/Programming/Chess Engine/Chess-Engine/Models/BlackModel_21_36(11)_RL(3)_selfplay_SGD.keras'  # Example for WSL
whiteModel.save(data_path1)
blackModel.save(data_path2)