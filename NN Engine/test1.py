# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 00:36:56 2024

@author: Kumodth
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 22:33:45 2024

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

    
    # EVASION TRAINING
    
    print ("Loading evasion training data...")
    
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
                if (inGameCount % 2 == 1):
                    
                    safety_moves = []
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
    
    # EVASION TRAINING
    
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
                if (inGameCount % 2 == 1):
                    
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
  

def lr_schedule(epoch, lr):
    if epoch == 0:
        lr = 0.0005 - trainingCount * 0.000005
    if epoch % 5 == 0 and epoch != 0:
        lr = lr * 0.5
    if lr <= 0.00005:
        lr = 0.00005
    return lr

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
    
   # Define the model
    inputs = tf.keras.Input(shape=(8, 8, 12))
    
    # Convolutional Layers
    x = Conv2D(filters=12, kernel_size=(3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.00001))(inputs)
    x = BatchNormalization()(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.00001))(x)
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
    
    x = transformer_block(x, num_heads=4, ff_dim=256)
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
    
    # Compile the model using Adam and loss as categorical crossentropy for classification of the moves
    initial_lr = 0.001  # Initial learning rate
    optimizer = Adam(learning_rate=initial_lr)
    #optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
    print(model.summary())

#  python3 /mnt/c/Users/Kumodth/Desktop/Programming/Chess\ Engine/Chess-Engine/NN\ Engine/WhiteNNTrainer.py