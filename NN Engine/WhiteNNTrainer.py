# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 22:33:45 2024

@author: Ranuja
"""

#Black File Creation
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, InputLayer, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import TopKCategoricalAccuracy
#from pickle import dump
from timeit import default_timer as timer
import chess
import chess.pgn
import io

board = chess.Board()
board.legal_moves

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
    
    # Open game data file
    pgn = open("../PGNs/LargeSet.pgn")
    
    t0= timer()
    
    # Holds the starting position for training
    gamePosition = 1
    
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
            if (inGameCount >= gamePosition and inGameCount <= 200):
                
                # If the board state is that white just moved, add the board as the input
                if (count % 2 == 1):
                    inputData.append(encode_board(board))
                    
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
                    print(temp[1204], reversePrediction(a,b,c,d), a,b,c,d)
                    output.append(temp) 
                    temp = [0]*4096
            
            # Increment the counts and make the move
            count += 1   
            inGameCount += 1
            board.push(move)
            

        else:
            # If the count is even, then there is an extra input without a corresponding output which should be removed  
            if (count % 2 == 0):
                count -= 1
                if(inGameCount >= gamePosition and inGameCount <= 200 and inGameCount % 2 == 0):
                    inputData.pop()
    
    # Convert the input and output into numpy arrays
    x = np.array([i for i in inputData])
    y = np.array([i for i in output])
    
    # Create the model
    model = Sequential()
    
    # Define the input as 12 channel 8*8 boards
    model.add(InputLayer(shape=(8, 8, 12)))
    
    # Convolutional Layers to identify patterns
    # Use kernel size of 3 as a sliding window
    # Utilize ReLU as activation
    
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same'))
    
    # Flatten the 3D output to 1D for the Dense layer
    model.add(Flatten())  
    
    # Fully connected layers with 256 and 128 neurons
    # Large capacity to capture complex patterns
    model.add(Dense(256, activation='relu'))  
    
    # Add dropout to reduce overfitting
    model.add(Dropout(0.5))  
    
    # Reduced capacity to condense features
    model.add(Dense(128, activation='relu'))  
    
    # Add dropout to reduce overfitting
    model.add(Dropout(0.5))  # Add dropout to reduce overfitting
    
    # The output uses softmax to define each output as a possible percentage
    model.add(Dense(4096,activation = 'softmax'))
    
    # Compile the model using Adam and loss as categorical crossentropy for classification of the moves
    model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=['accuracy', TopKCategoricalAccuracy(k=5)])
    
    # Implement Early Stopping
    early_stopping = EarlyStopping(
        monitor='val_loss',  # Metric to monitor
        patience=5,          # Number of epochs with no improvement after which training will be stopped
        restore_best_weights=True  # Restore the model weights from the epoch with the best value of the monitored quantity
    )
    
    # Train the model with early stopping
    history = model.fit(
        x, y,
        epochs=50,  # Set a large number of epochs for the possibility of early stopping
        batch_size=32,
        validation_split=0.2,  # Split a portion of the data for validation
        callbacks=[early_stopping],  # Pass the early stopping callback
        shuffle=True,
        verbose=1
        )
    
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
    model.save('../Models/WhiteModel1.keras')
    t1 = timer()
    print("Time elapsed: ", t1 - t0)
    

