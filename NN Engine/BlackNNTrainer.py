# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 14:30:31 2024

@author: Ranuja
"""

#Black File Creation
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, InputLayer, Dropout, BatchNormalization, GlobalAveragePooling2D, MaxPooling2D, MultiHeadAttention, Add, LayerNormalization
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from tensorflow.keras.metrics import TopKCategoricalAccuracy
from tensorflow.keras.optimizers import Adam, Nadam, RMSprop
from timeit import default_timer as timer
import chess
import chess.pgn
import io
import platform
import gc
import math
from numba import cuda
import itertools
from itertools import zip_longest

board = chess.Board()
board.legal_moves

trainingCount = 0
gameStart = 21
gameUntil = 36

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

def evasionTraining():
    
    # CHECKMATE TRAINING
    
    temp = [0]*4096
    inputData = []
    output = []
    
    # Open game data file
    if platform.system() == 'Windows':
        data_path = r'../PGNs/blackWins.pgn'
    elif platform.system() == 'Linux':
        data_path = '/mnt/c/Users/Kumodth/Desktop/Programming/Chess Engine/Chess-Engine/PGNs/LargeSet.pgn'
    pgn = open(data_path)
    
    gamePosition = gameStart
    
    # Iterate through all moves and play them on a board.
    count = 1
    with tf.device('/cpu:0'):
        while True:
            # Exit the loop when the game file has reached its end
            game = chess.pgn.read_game(pgn)
            if game is None:
                print("asd")
                break
            
            # Set the game board to the starting position and move count to 1
            board = game.board()
            
            inGameCount = 1
            for move in game.mainline_moves():
        
                # Check if the game count is within the training bounds
                if (inGameCount >= gamePosition and inGameCount <= gameUntil):
                    
                    # If the board state is that white just moved, add the board as the input
                    if (inGameCount % 2 == 0):
                        
                        for legalMove in board.legal_moves:
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
                                    
                                    board.pop()
                                    
                                    inputData.append(encode_board(board))
                                    inputData.append(encode_board(reflect_board(board)))
                                    '''
                                    print("~~~~~~~~~~~~~~~~~~~~~")
                                    print(board)
                                    print()
                                    '''
                                    board.push(legalMove)
                                    '''
                                    print(board)
                                    print("~~~~~~~~~~~~~~~~~~~~~")
                                    display_board(board, label=f"Move Made: {not board.turn}")
                                    '''
                                board.pop()
        
                
                # Increment the counts and make the move
                count += 1   
                inGameCount += 1
                board.push(move)
            
    print(count)        
    return inputData, output 

def captureTraining():
    
    # CAPTURE TRAINING
    
    
    print ("Loading capture training data...")
    
    temp = [0]*4096
    inputData = []
    output = []
    
    # Open game data file
    if platform.system() == 'Windows':
        data_path = r'../PGNs/blackWins.pgn'
    elif platform.system() == 'Linux':
        data_path = '/mnt/c/Users/Kumodth/Desktop/Programming/Chess Engine/Chess-Engine/PGNs/LargeSet.pgn'
    pgn = open(data_path)
    
    gamePosition = gameStart
    
    # Iterate through all moves and play them on a board.
    count = 1
    while True:
        
        # Exit the loop when the game file has reached its end
        game = chess.pgn.read_game(pgn)
        if game is None:
            print("asd")
            break
        
        # Set the game board to the starting position and move count to 1
        board = game.board()
        
        inGameCount = 1
        for move in game.mainline_moves():
    
            # Check if the game count is within the training bounds
            if (inGameCount >= gamePosition and inGameCount <= gameUntil):
                
                # If the board state is that white just moved, add the board as the input
                if (inGameCount % 2 == 0):
                    
                    safety_moves = []
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
                            '''
                            print("~~~~~~~~~~~~~~~~~~~~~")
                            print(board)
                            print()
                            
                            board.push(legalMove)
                            
                            print(board)
                            print("~~~~~~~~~~~~~~~~~~~~~")
                            #display_board(board, label=f"Move Made: {not board.turn}")
                            
                            board.pop()
                            '''
            
            # Increment the counts and make the move
            count += 1   
            inGameCount += 1
            board.push(move)
            
    print(count)        
    return inputData, output 
  
def transformer_block(inputs, num_heads, ff_dim):
    # Multi-Head Attention
    attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=inputs.shape[-1])(inputs, inputs)
    attention_output = Add()([inputs, attention_output])
    attention_output = LayerNormalization()(attention_output)
    
    # Feed-Forward Network
    ff_output = Dense(ff_dim, activation='relu')(attention_output)
    ff_output = Dense(inputs.shape[-1])(ff_output)
    ff_output = Add()([attention_output, ff_output])
    ff_output = LayerNormalization()(ff_output)
    
    return ff_output

def lr_schedule(epoch, lr):
    if epoch == 0:
        lr = 0.0005 - trainingCount * 0.000005
    if epoch % 3 == 0 and epoch != 0:
        lr = lr * 0.5
    if lr <= 0.00005:
        lr = 0.00005
    return lr

if __name__ == "__main__":    
    
    # Define variables to create the training data
    temp = [0]*4096
    inputData = []
    output = []
    
    # Create the test board
    newPgn = io.StringIO("1. e4*")
    newGame = chess.pgn.read_game(newPgn)
    testBoard = newGame.board()
    for move in newGame.mainline_moves():
        pass
    testBoard.push(move.from_uci('e2e4'))
        
    t0 = timer()
    
    # Open game data file
    if platform.system() == 'Windows':
        data_path = r'../PGNs/SuperSet.pgn'
    elif platform.system() == 'Linux':
        data_path = '/mnt/c/Users/Kumodth/Desktop/Programming/Chess Engine/Chess-Engine/PGNs/SuperSet.pgn'  # Example for WSL
    pgn = open(data_path)
    
    # Holds the starting position for training
    gamePosition = gameStart
    
    # Iterate through all moves and play them on a board.
    count = 1
    while True:
        
        # Exit the loop when the game file has reached its end
        game = chess.pgn.read_game(pgn)
        if game is None:
            print("asd")
            break
        
        # Set the game board to the starting position and move count to 1
        board = game.board()
        inGameCount = 1
        for move in game.mainline_moves():
            
            # Make the move
            board.push(move)
            
            # Check if the game count is within the training bounds
            if (inGameCount >= gamePosition and inGameCount <= gameUntil):
                
                if (count % 2 == 1):
                    inputData.append(encode_board(board))
                    inputData.append(encode_board(reflect_board(board)))
                # If black just made a move, make that move the output of the input
                else:
    
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
            
            # Increment the counts
            count += 1   
            inGameCount += 1
            
        else:
            # If the count is even, then there is an extra input without a corresponding output which should be removed  
            if (count % 2 == 0):
                count -= 1
                if(inGameCount >= gamePosition and inGameCount <= gameUntil and inGameCount % 2 == 0):
                    inputData.pop()
                    inputData.pop()
    
    
    cuda.select_device(0)
    cuda.current_context().reset()
    a,b = evasionTraining()    
    inputData = [item for pair in zip_longest(inputData, a) for item in pair if item is not None]
    output = [item for pair in zip_longest(output, b) for item in pair if item is not None]
    
    del a,b
    gc.collect()
    
    a,b = captureTraining()    
    inputData = [item for pair in zip_longest(inputData, a) for item in pair if item is not None]
    output = [item for pair in zip_longest(output, b) for item in pair if item is not None]
    
    del a,b
    gc.collect()
    
    # Define the model
    inputs = tf.keras.Input(shape=(8, 8, 12))
    
    # Convolutional Layers
    x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.00001))(inputs)
    x = BatchNormalization()(x)
    x = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.00001))(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.00001))(x)
    x = BatchNormalization()(x)
    #x = MaxPooling2D(pool_size=(2, 2))(x)
    # x = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.00001))(x)
    # x = BatchNormalization()(x)
    # x = MaxPooling2D(pool_size=(2, 2))(x)
    # x = Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.00001))(x)
    # x = BatchNormalization()(x)
    # x = MaxPooling2D(pool_size=(2, 2))(x)
    # x = Conv2D(filters=512, kernel_size=(4, 4), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.00001))(x)
    # x = BatchNormalization()(x)
    # x = MaxPooling2D(pool_size=(2, 2))(x)
    # x = Conv2D(filters=1024, kernel_size=(4, 4), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.0001))(x)
    # x = BatchNormalization()(x)
    
    #x = transformer_block(x, num_heads=4, ff_dim=64)
    # Global Average Pooling
    #x = GlobalAveragePooling2D()(x)
    x = Flatten()(x)
    # Fully connected layers
    x = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.00001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.05)(x)
    x = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.00001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.05)(x)
    
    # Output layer
    outputs = Dense(4096, activation='softmax')(x)
    
    # Create and compile the model
    model = tf.keras.Model(inputs, outputs)     
    
    
    # Define the learning rate scheduler
    initial_lr = 0.001  # Initial learning rate
    optimizer = Adam(learning_rate=initial_lr)
    
    print(model.summary())
    num_samples = len(inputData)
    print("Input Size: ", len(inputData))
    #count = 0
    trainingCount = 0
    
    for i in range (3):
        print ("Iteration:", i)
        for start_idx in range(0, num_samples, 100000):
            end_idx = min(start_idx + 100000, num_samples)
            
            # Convert the input and output into numpy arrays
            x = np.array(inputData[start_idx:end_idx])
            y = np.array(output[start_idx:end_idx])
            
            cuda.select_device(0)
            cuda.current_context().reset()
            #K.set_value(model.optimizer.learning_rate, new_lr)
            print("Starting Batch:",trainingCount+1, "From index:",start_idx, "to:", end_idx,'\n')
            
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
            trainingCount+=1
            print("Done:",trainingCount)
            

            del x, y
            gc.collect()

    print(model.summary())
    
    # Make a prediction using the test board
    q = model.predict(np.array([encode_board(testBoard)]))
    
    print(np.argmax(q))    
    a,b,c,d = predictionInfo(np.argmax(q))
    print(len(q))
    print("X1: ",a)
    print("Y1: ",b)
    print("X2: ",c)
    print("Y2: ",d)
    
    print(reversePrediction(a,b,c,d))
    
    # Save the model
      
    if platform.system() == 'Windows':
        data_path = r'../Models/BlackModel6_MidEndGame(9).keras'
    elif platform.system() == 'Linux':
        data_path = '/mnt/c/Users/Kumodth/Desktop/Programming/Chess Engine/Chess-Engine/Models/BlackModel_21_36(6).keras'  # Example for WSL
    model.save(data_path)

    t1 = timer()
    print("Time elapsed: ", t1 - t0)