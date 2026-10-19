# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 17:13:58 2024

@author: Kumodth
"""

from timeit import default_timer as timer
#from numba import njit
import easygui
import copy
import Rules
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, InputLayer, Dropout, BatchNormalization, ReLU, GlobalAveragePooling2D, MaxPooling2D, MultiHeadAttention, Add, LayerNormalization
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from tensorflow.keras.metrics import TopKCategoricalAccuracy
from tensorflow.keras.optimizers import Adam, Nadam, RMSprop
#from pickle import dump
from timeit import default_timer as timer
import chess
import chess.pgn
import io
import platform
import os
from tensorflow.keras.models import Model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Set TensorFlow log level to suppress all but errors

pgnBoard = chess.Board()
pgnBoard.legal_moves

if platform.system() == 'Windows':
    data_path1 = '../Models/BlackModel_21_36(11)_selfplay_SGD.keras'
    data_path2 = '../Models/WhiteModel_21_36(11)_selfplay_SGD.keras'
elif platform.system() == 'Linux':
    data_path1 = '/mnt/c/Users/Kumodth/Desktop/Programming/Chess Engine/Chess-Engine/Models/BlackModel_21_36(11)_selfplay_SGD.keras'
    data_path2 = '/mnt/c/Users/Kumodth/Desktop/Programming/Chess Engine/Chess-Engine/Models/WhiteModel_21_36(11)_selfplay_SGD.keras'
    

blackModel = tf.keras.models.load_model(data_path1)
whiteModel = tf.keras.models.load_model(data_path2)

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

def generalTraining():
    
    # GENERAL TRAINING
    
    print ("Loading general training data...")
    
    temp = [0]*4096
    inputData = []
    output = []
        
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
    
    
    return inputData, output

def lr_schedule(epoch, lr):
    if epoch == 0:
        lr = 0.0005
    if epoch % 3 == 0 and epoch != 0:
        lr = lr * 0.5
    if lr <= 0.00000005:
        lr = 0.00000005
    return lr

t0 = timer()
inputData, output = generalTraining()
inputData = np.array([inputData])
output = np.array(output)
# Extract features from an intermediate layer
# optimizer = Adam(learning_rate=0.001)
# blackModel.compile(optimizer=optimizer, loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.2), metrics=['accuracy', TopKCategoricalAccuracy(k=10)])

#blackModel.predict(np.array([inputData[0]]))  # This will initialize the input shape
feature_layer_model = Model(inputs = blackModel.get_layer('input_layer').output, outputs=blackModel.get_layer('dense_1').output)  # Adjust the layer name

extracted_features = []

for inp in inputData:
    print(feature_layer_model.predict(inp))
    extracted_features.append(feature_layer_model.predict(inp))  # X_data is your chess board data

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Combine features with additional features if necessary
#combined_features = extracted_features  # You can concatenate additional features here

print(extracted_features, output)
# Split the combined features into training and testing sets
X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(extracted_features[0], output, test_size=0.2, random_state=42)

# Initialize and train the Random Forest model
rf_model = RandomForestClassifier(n_estimators=10, random_state=42, verbose=3)
rf_model.fit(X_train_rf, y_train_rf)

# Evaluate the Random Forest model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

y_pred_rf = rf_model.predict(X_test_rf)
rf_accuracy = accuracy_score(y_test_rf, y_pred_rf)
print(f"Random Forest Accuracy: {rf_accuracy:.2f}")
#print(classification_report(y_test_rf, y_pred_rf))
#print(confusion_matrix(y_test_rf, y_pred_rf))
t1 = timer()
print("Time elapsed: ", t1 - t0)
import pickle

# Save the trained Random Forest model to a file
with open('random_forest_model.pkl', 'wb') as model_file:
    pickle.dump(rf_model, model_file)
