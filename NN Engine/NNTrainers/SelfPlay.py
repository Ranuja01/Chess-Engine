"""
@author: Ranuja Pinnaduwage

This file allows the engine to play itself and generate data for back propagation training

"""

import chess
import chess.engine
import chess.pgn
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, InputLayer, Dropout, BatchNormalization, GlobalAveragePooling2D, MaxPooling2D, MultiHeadAttention, Add, LayerNormalization
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from tensorflow.keras.metrics import TopKCategoricalAccuracy
from tensorflow.keras.optimizers import Adam, Nadam, RMSprop
from timeit import default_timer as timer
from tensorflow.keras.losses import KLDivergence

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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Set TensorFlow log level to suppress all but errors

tf.config.optimizer.set_jit(True)  # Enable XLA

trainingCount = 0
loop = True

# Create a mutex lock and define an event (semaphore)
lock = threading.Lock()
event = threading.Event()

# Define the limit to how many entries can be used per training iteration
dataLimit = 100000

# Define the training data lists globally
black_inputData = []
black_output = []

white_inputData = []
white_output = []

if platform.system() == 'Windows':
    data_path1 = r'../Models/BlackModel_21_36.keras'
    data_path2 = r'../Models/WhiteModel_21_36.keras'

elif platform.system() == 'Linux':
    
    data_path1 = r'/mnt/c/Users/Kumodth/Desktop/Programming/Chess Engine/Chess-Engine/Models/BlackModel_21_36(12)_RL(1).keras'
    data_path2 = r'/mnt/c/Users/Kumodth/Desktop/Programming/Chess Engine/Chess-Engine/Models/WhiteModel_21_36(12)_RL(1).keras'
    
blackModel = tf.keras.models.load_model(data_path1)
whiteModel = tf.keras.models.load_model(data_path2)

# Set the path to the Stockfish binary
if platform.system() == 'Windows':
    STOCKFISH_PATH = "../../stockfish/stockfish-windows-x86-64-avx2"  # Make sure this points to your Stockfish binary
elif platform.system() == 'Linux':
    STOCKFISH_PATH = "/mnt/c/Users/Kumodth/Desktop/Programming/Chess Engine/stockfish/stockfish-windows-x86-64-avx2.exe"  # Make sure this points to your Stockfish binary

def is_promotion_move_enhanced(move, board):
    
    """
    Function that checks if a move is a promotion move.

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
    Function that Suggests the best move(s) for the given position.

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
    Function to get Stockfish evaluation of the given board position.

    Parameters:
    - board: chess.Board object
    - engine: chess.engine.SimpleEngine instance
    - time_limit: float, time to analyze the position in seconds

    Returns:
    - evaluation: float, centipawn evaluation or large numerical value for mate
    """
    
    result = engine.analyse(board, chess.engine.Limit(time=time_limit))
    score = result["score"].white()

    # Convert Mate in X to a large numerical value
    if score.is_mate():
        mate_in_moves = score.mate()
        evaluation = 1000 * (1 - mate_in_moves/200) if mate_in_moves > 0 else -1000 * (1 - mate_in_moves/200)
    else:
        evaluation = score.score(mate_score=10000) / 100.0
    
    return evaluation

def evaluate_pgn(file_path, stockfish_path=STOCKFISH_PATH, time_limit=0.01):
    
    """
    Function to analyze a PGN file and print Stockfish evaluations for each move.
    
    Parameters:
    - file_path: str, path to the PGN file
    - stockfish_path: str, path to the Stockfish engine
    - time_limit: float, time to analyze each position in seconds
    """
    
    # Open the PGN file
    with open(file_path) as pgn_file:
        # Read the first game from the PGN file
        game = chess.pgn.read_game(pgn_file)
        
        # Initialize the board
        board = game.board()
        
        # Start the Stockfish engine
        with chess.engine.SimpleEngine.popen_uci(stockfish_path) as engine:
            # Go through each move in the game
            move_number = 1
            for move in game.mainline_moves():
                print(move)
                board.push(move)  # Play the move
                
                # Get evaluation after the move
                evaluation = get_stockfish_evaluation(board, engine, time_limit)
                
                print("Current position:\n", board)
                print()
                # Print the move and its evaluation
                print(f"Move {move_number}: {move}, Evaluation: {evaluation}")                
                
                moveMade = str(move)
                a = ord(moveMade[0:1]) - 96
                b = int(moveMade[1:2])
                c = ord(moveMade[2:3]) - 96
                d = int(moveMade[3:4])
                
                print(a,b,c,d)
                # Get move suggestions                
                suggestions = suggest_moves(board, engine, time_limit=0.001, depth=20, multipv=2)
                print()
                # Display the top suggested moves
                print("\nTop suggested moves:")
                for i, move in enumerate(suggestions):
                    print(f"{i+1}. Move: {board.san(move)}")
                                
                move_number += 1

def getNNMove(board):
    
    """
    Function to get move from the neural network

    Parameters:
    - board: chess.Board object

    Returns:
    - chess.Move from the neural network
    """
    
    # Define a list for filtering the neural network output to legal moves 
    filteredPrediction = [0]*4096
    
    # Call the function to convert the current board to a 12 channel tensor
    inputBoard = [encode_board(board)]
    if board.turn:
        
        # Acquire the probability distribution from the neural network
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
        
        # Call the function to convert the prediction to coordinates
        a,b,c,d = predictionInfo(np.argmax(filteredPrediction))
        
    else:
    
        # Acquire the probability distribution from the neural network
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
        
        # Call the function to convert the prediction to coordinates
        a,b,c,d = predictionInfo(np.argmax(filteredPrediction))

    # Convert the coordinates to algebraic notation
    a = chr(a + 96)
    b = str(b)
    c = chr(c + 96)
    d = str(d)
    
    # If no move is chosen because all are equally considered bad, then return a random move
    if ((a,b,c,d) == ('a','1','a','1')):
        return getRandomMove(board)
    
    # If a move is a promotion, default to a queen
    if (is_promotion_move_enhanced(move.from_uci(a+b+c+d),board)):
        return move.from_uci(a+b+c+d+'q')
    else:
        return move.from_uci(a+b+c+d)

def getRandomMove(board):
    
    """
    Function to get a random move

    Parameters:
    - board: chess.Board object

    Returns:
    - A random chess.Move
    """
    
    legal_moves = list(board.legal_moves)    
    return random.choice(legal_moves) if legal_moves else None

def lr_schedule_adam(epoch, lr):
    
    """
    Function to get the learning rate schedule specifically for the adam optimizer

    Parameters:
    - epoch: int, represents the current training iteration
    - lr: float, represents the current learning rate 

    Returns:
    - floating point learning rate
    """
    
    if epoch == 0:
        lr = 0.0005 - trainingCount * 0.00005
    if epoch % 3 == 0 and epoch != 0:
        lr = lr * 0.5
    if lr <= 0.00000005:
        lr = 0.00000005
    return lr

def lr_schedule(epoch, lr):
    
    """
    Function to get the learning rate schedule

    Parameters:
    - epoch: int, represents the current training iteration
    - lr: float, represents the current learning rate 

    Returns:
    - floating point learning rate
    """
    
    if epoch == 0:
        lr = 0.05 - trainingCount * 0.007
    if epoch % 2 == 0 and epoch != 0:
        lr = lr * 0.5
    if lr <= 0.000005:
        lr = 0.000005
    return lr

def trainModel(model, inputData, output):
    
    """
    Function to train the given model

    Parameters:
    - model: Tensorflow model, to be trained
    - inputData: A list of 12 channel tensors
    - output: A list of probability distributions

    """
    
    # Define the model optimizer
    #optimizer = Adam(learning_rate=initial_lr)
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.005, momentum=0.9)
    num_samples = len(inputData)
    print("Input Size: ", len(inputData))
    
    # Loop through the training data multiple times    
    for i in range (1):
        print ("Iteration:", i)
        
        # Partition the training data so as to not overfill the GPU
        for start_idx in range(0, num_samples, 100000):
            end_idx = min(start_idx + 100000, num_samples)
            
            # Convert the input and output into numpy arrays
            x = np.array(inputData[start_idx:end_idx])
            y = np.array(output[start_idx:end_idx])
            
            # Reset the memory of the GPU
            cuda.select_device(0)
            cuda.current_context().reset()
            
            print("Starting Batch:",trainingCount+1, "From index:",start_idx, "to:", end_idx,'\n')
            
            # Define the learning rate scheduler and compile the model
            # Use Categorical cross entropy for the probability distribution with label smoothing.
            lr_scheduler = LearningRateScheduler(lr_schedule)
            model.compile(optimizer=optimizer, loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.2), metrics=['accuracy', TopKCategoricalAccuracy(k=10)])
            
            # Implement Early Stopping
            early_stopping = EarlyStopping(
                monitor='val_top_k_categorical_accuracy',  # Metric to monitor
                patience=6,          # Number of epochs with no improvement after which training will be stopped
                restore_best_weights=True  # Restore the model weights from the epoch with the best value of the monitored quantity
            )
            
            # Train the model with early stopping
            history = model.fit(
                x, y,
                epochs=50,  # Set a large number of epochs for the possibility of early stopping
                batch_size=64,
                validation_split=0.2,  # Split a portion of the data for validation
                callbacks=[lr_scheduler, early_stopping],  # Pass the early stopping callback and learning rate scheduler
                shuffle=True,
                verbose=1
                )
            #trainingCount+=1
            print("Done:",trainingCount)
            
            # Delete the input and output data and call the garbage collector
            del x, y
            gc.collect()
            
    print(model.summary())

def appendData(board, inputData, output, engine):
    
    """
    Function to append the current input and output data to the total

    Parameters:
    - board: chess.Board object, that holds the current board state
    - inputData: A list of 12 channel tensors
    - output: A list of probability distributions
    - engine: an instance of the stockfish engine

    """
    
    temp = [0]*4096
    
    move = board.pop()
    inputData.append(encode_board(board))
    inputData.append(encode_board(reflect_board(board)))
    board.push(move)
    # Convert the move into coordinates
    moveMade = str(board.peek())
    a = ord(moveMade[0:1]) - 96
    b = int(moveMade[1:2])
    c = ord(moveMade[2:3]) - 96
    d = int(moveMade[3:4])
    
    # Set the index of the output corresponding to the 4 coordinates as 100%
    temp [reversePrediction(a,b,c,d) - 1] = 1
    
    output.append(temp) 
    temp = [0]*4096
    
    # Set the index of the output corresponding to the 4 coordinates as 100%
    temp [reversePrediction(9 - a,b,9 - c,d) - 1] = 1
    
    output.append(temp) 
    temp = [0]*4096
    
    board.pop()
    moves = suggest_moves(board, engine, time_limit=0.001, depth=15, multipv=5)
    for stockfish_move in moves[1:]:
        
        inputData.append(encode_board(board))
        inputData.append(encode_board(reflect_board(board)))
        board.push(stockfish_move)
        
        # Convert the move into coordinates
        moveMade = str(board.peek())
        a = ord(moveMade[0:1]) - 96
        b = int(moveMade[1:2])
        c = ord(moveMade[2:3]) - 96
        d = int(moveMade[3:4])
        
        # Set the index of the output corresponding to the 4 coordinates as 100%
        temp [reversePrediction(a,b,c,d) - 1] = 1
        
        output.append(temp) 
        temp = [0]*4096
        
        # Set the index of the output corresponding to the 4 coordinates as 100%
        temp [reversePrediction(9 - a,b,9 - c,d) - 1] = 1
        
        output.append(temp) 
        temp = [0]*4096
        board.pop()
    board.push(move)
    
    # ** Extra code to include capture moves and evasive moves
    
    '''
    board.pop()
    for legalMove in board.legal_moves:
        
        if board.is_capture(legalMove):
            
            board.push(legalMove)
                
            # Convert the move into coordinates
            moveMade = str(board.peek())
            a = ord(moveMade[0:1]) - 96
            b = int(moveMade[1:2])
            c = ord(moveMade[2:3]) - 96
            d = int(moveMade[3:4])
            
            # Set the index of the output corresponding to the 4 coordinates as 100%
            temp [reversePrediction(a,b,c,d) - 1] = 1
            #print(temp[1204], reversePrediction(a,b,c,d), a,b,c,d)
            output.append(temp) 
            temp = [0]*4096
            
            # Set the index of the output corresponding to the 4 coordinates as 100%
            temp [reversePrediction(9 - a,b,9 - c,d) - 1] = 1
            #print(temp[1204], reversePrediction(a,b,c,d), a,b,c,d)
            output.append(temp) 
            temp = [0]*4096
            
            board.pop()
            
            inputData.append(encode_board(board))
            inputData.append(encode_board(reflect_board(board)))
        
        from_square = legalMove.from_square
        to_square = legalMove.to_square
        piece_under_attack = board.is_attacked_by(not board.turn, from_square)

        if piece_under_attack:
            
            board.push(legalMove)
            # Check if the destination square is not attacked by the opponent
            if not board.is_attacked_by(board.turn, to_square):
                
                # Convert the move into coordinates
                moveMade = str(board.peek())
                a = ord(moveMade[0:1]) - 96
                b = int(moveMade[1:2])
                c = ord(moveMade[2:3]) - 96
                d = int(moveMade[3:4])
                
                # Set the index of the output corresponding to the 4 coordinates as 100%
                temp [reversePrediction(a,b,c,d) - 1] = 1
                #print(temp[1204], reversePrediction(a,b,c,d), a,b,c,d)
                output.append(temp) 
                temp = [0]*4096
                
                # Set the index of the output corresponding to the 4 coordinates as 100%
                temp [reversePrediction(9 - a,b,9 - c,d) - 1] = 1
                #print(temp[1204], reversePrediction(a,b,c,d), a,b,c,d)
                output.append(temp) 
                temp = [0]*4096
                
                inputData.append(encode_board(board))
                inputData.append(encode_board(reflect_board(board)))
                
            board.pop()
    board.push(move)
    '''

def predictionInfo(prediction):
    
    """
    Function to convert the neural network output to 4 coordinates

    Parameters:
    - prediction: An integer index

    Returns:
    - A tuple consisting of the 4 coordinates
    """
    
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

def reversePrediction(x,y,i,j):
    
    """
    A function that converts the coordinates back into the NN output (probability distribution)

    Parameters:
    - x: int, the starting x coordinate
    - y: int, the starting y coordinate
    - i: int, the destination x coordinate
    - j: int, the destination y coordinate

    Returns:
    - A tuple consisting of the 4 coordinates
    """
    
    # First acquire the starting square number and multiply by 64 to get its base number
    # Then add the remaining starting point of the location to be moved to
    return (((x - 1) * 8 + y) - 1)  *64 + ((i - 1) * 8 + j)
                  
def encode_board(board):
    
    """
    A function that converts the board into a 12 channel tensor    

    Parameters:
    - board: chess.Board, the current board state

    Returns:
    - A 12 channel tensor where only one type of piece exists per channel
    """
    
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
    
    """
    A function that reflects the board across the verticl axis

    Parameters:
    - board: chess.Board, the current board state

    Returns:
    - The reflected chess.Board object
    """
    
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

def selfPlay():
    
    """
    A function where a game is played using with the NN to generate training data
    """
    
    stockfish_path=STOCKFISH_PATH
    
    # Acquire the global loop variable for when the training data has reached capacity for training
    global loop
    
    # Define the part of the game from which the training data should be acquired
    gameStart = 21
    gameUntil = 36
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    
    for i in range (2):
        
        # Keep looping until the global variable suggests training data has reached capacity
        while(loop):
            # current_thread = threading.current_thread()
            # Print the thread ID
            #print(f"Thread Name: {current_thread.name}, Thread ID: {current_thread.ident}")
            #print("Black Size: ",len(black_output))    
            #print("White Size: ",len(white_output))    
        
            board = chess.Board()
            inGameCount = 1
                        
            # Reset arrays
            black_inputData_temp = []
            black_output_temp = []
            
            white_inputData_temp = []
            white_output_temp = []  
            
            # Loop through a game
            while(not(board.is_game_over())):
            
                # Generate a random number between 0 and 1
                random_number = random.random()
                
                stockfish_usage = 0.5
                
                # If one side is very slow in acquiring training data, increase the usage of stockfish to speed up
                if (len(black_inputData) > dataLimit and board.turn) or (len(white_inputData) > dataLimit and not(board.turn)):
                    stockfish_usage = 1.0
                                
                # Define the probability thresholds for each branch
                if random_number < stockfish_usage:  # 50% chance for the first branch
                    
                    # Call stockfish to get a move
                    move = suggest_moves(board, engine, time_limit=0.001, depth=15, multipv=1)
                    board.push(move[0])                   
                
                elif random_number < 0.95:  # 45% chance for the second branch
                    
                    # Call the neural network to get a move
                    move = getNNMove(board)
                    board.push(move)
                    
                else:  # 5% chance for the third branch
                    
                    # Call the randomizer function to get a random move
                    move = getRandomMove(board)
                    board.push(move)
     
                if (inGameCount > gameUntil) or board.is_game_over():
                   
                    # Use the stockfish evaluation of the given board state to determine if this game should be appended to white, black or both
                    evaluation = get_stockfish_evaluation(board, engine, 0.001)
                    
                    if (evaluation >= -0.90 and evaluation <= 0.85):
                        
                        if len(black_inputData) <= dataLimit + 50000:
                            black_inputData.extend(black_inputData_temp)
                            black_output.extend(black_output_temp)
                        
                        if len(white_inputData) <= dataLimit + 50000:
                            white_inputData.extend(white_inputData_temp)
                            white_output.extend(white_output_temp)
                                            
                    elif (evaluation < -0.90):
                        if len(black_inputData) <= dataLimit + 50000:
                            black_inputData.extend(black_inputData_temp)
                            black_output.extend(black_output_temp)
                        
                    elif (evaluation > 0.85):
                        if len(white_inputData) <= dataLimit + 50000:
                            white_inputData.extend(white_inputData_temp)
                            white_output.extend(white_output_temp)                                        
                    break
    
                # Within the appropriate range, begin appending the appropriate data
                # Use mutex lock to keep the data aligned properly when using multiple threads
                if (inGameCount >= gameStart):
                    
                    if board.turn:            
                        with lock:                            
                            appendData(board, white_inputData_temp, white_output_temp, engine)                        
                    else:                        
                        with lock:
                            appendData(board, black_inputData_temp, black_output_temp,engine)
                        
                inGameCount += 1
                    
            del black_inputData_temp, black_output_temp, white_inputData_temp, white_output_temp
            #gc.collect()
            
        # Each thread should wait until signalled when the training is over
        event.wait() 
        loop = True
        event.clear() 
    engine.close()   

# List to hold the thread objects
threads = []

# Create and start threads
for i in range(12):  # Example with 5 threads
    t = threading.Thread(target=selfPlay, args=())
    t.start()
    threads.append(t)

# Main function continues to execute
print("Main function is doing other things...")

# Wait for threads to complete
count = 0

t0_full = timer()
t0 = timer()

# Loop while any thread is still alive
while any(t.is_alive() for t in threads):
    #print("Waiting for threads to finish...")
    time.sleep(5)
    loop = True
    
    # Check if the training data has reached capacity
    if len(white_output) > dataLimit and len(black_output) > dataLimit:
        t1 = timer()
        print("Time elapsed: ", t1 - t0)
        print("Copying...")
        
        # Stop all threads from adding new data
        loop = False
        
        # Copy the data so that the original lists can be emptied
        blackIn = copy.deepcopy(black_inputData)
        blackOut = copy.deepcopy(black_output)
        
        whiteIn = copy.deepcopy(white_inputData)
        whiteOut = copy.deepcopy(white_output)
                
        #del black_inputData, black_output, white_inputData, white_output
        #gc.collect()        
        
        
        # Empty global lists
        black_inputData = []
        black_output = []

        white_inputData = []
        white_output = []
        
        print("Done!")
        
        
        # Call the training function to train each model
        trainModel(whiteModel, whiteIn, whiteOut)
        trainModel(blackModel, blackIn, blackOut)
        
        # Signal the continuation of collecting training data
        event.set()
        
        # Delete the data copies and call the garbage collector
        del blackIn, blackOut, whiteIn, whiteOut
        gc.collect()
        
        t0 = timer()
        trainingCount += 1
    count+=1
print("All threads have finished.")
t1_full = timer()
print("Time elapsed: ", t1_full - t0_full)

# Save the models
if platform.system() == 'Windows':
    data_path = r'../Models/WhiteModel6_MidEndGame(8)_Refined.keras'
elif platform.system() == 'Linux':
    data_path1 = '/mnt/c/Users/Kumodth/Desktop/Programming/Chess Engine/Chess-Engine/Models/WhiteModel_21_36(12)_RL(1)_selfplay_SGD.keras'  # Example for WSL
    data_path2 = '/mnt/c/Users/Kumodth/Desktop/Programming/Chess Engine/Chess-Engine/Models/BlackModel_21_36(12)_RL(1)_selfplay_SGD.keras'  # Example for WSL
whiteModel.save(data_path1)
blackModel.save(data_path2)
    