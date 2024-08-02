# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 12:15:45 2024

@author: Ranuja
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 21:08:39 2024

@author: Ranuja
"""
from GUI import Pieces
from timeit import default_timer as timer
#from numba import njit
import easygui
import copy
import Rules
import numpy as np
#import tensorflow as tf
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Conv2D, Flatten, Dense, InputLayer
#from pickle import dump
from timeit import default_timer as timer
import chess
import chess.pgn
import io
import platform
import os
#import chess_eval
from ChessAI import ChessAI



os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Set TensorFlow log level to suppress all but errors

pgnBoard = chess.Board()

'''
pgnBoard.legal_moves
if platform.system() == 'Windows':
    data_path1 = '../Models/BlackModel4.keras'
    data_path2 = '../Models/WhiteModel1.keras'
elif platform.system() == 'Linux':
    data_path1 = '/mnt/c/Users/Kumodth/Desktop/Programming/Chess Engine/Chess-Engine/Models/BlackModel4.keras'
    data_path2 = '/mnt/c/Users/Kumodth/Desktop/Programming/Chess Engine/Chess-Engine/Models/WhiteModel1.keras'
    

blackModel = tf.keras.models.load_model(data_path1)
whiteModel = tf.keras.models.load_model(data_path2)
'''
chess_ai = ChessAI(3, 3, pgnBoard)


newPgn = io.StringIO("1. e4*")
newGame = chess.pgn.read_game(newPgn)

for move in newGame.mainline_moves():
    pass

def evaluateBoard(self):
        sum = 0

        #print ('\n')
        '''
        if(self.blackHasCastled):
            sum += 1500
        if(self.whiteHasCastled):
            sum -= 1500
        '''
        '''
        if(Rules.isCheckMate(chess.WHITE)):
            print("AAAAA")
            sum = 10000000
        elif(Rules.isCheckMate(chess.BLACK)):
            print("BBBB")
            sum = -10000000
        '''
        # Check for center square control
        for item in self.boardPieces:
            for square in item:
                if(not(square.piece == "King")):
                    
                    if (square.colour == "Black"):
                        sum += square.value
                    elif (square.colour == "White"):
                        sum -= square.value   

        
        return sum      

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

def engineMove(self):
        
    t0= timer()
    
    # Set variable such that certain features know that an actual move is not being attempted
    self.computerThinking = True
    
    # Call the alpha beta algorithm to make a move decision
    result = chess_ai.alphaBetaWrapper(curDepth=0, depthLimit=3)
    a,b,c,d = result['a'],result['b'],result['c'],result['d']
    val = result['score']
    #a,b,c,d,val = alphaBeta(self, 0,self.depth)
    self.computerThinking = False
    
    # If the algorithm does not select a move, that suggests they are all equally bad and therefore the machine resigns
    if(not(self.boardPieces[a - 1][b - 1] == None or self.boardPieces[a - 1][b - 1].piece == "Empty" or reversePrediction(a,b,c,d) == 1 or (a,b,c,d) == (-1,-1,-1,-1))):
        print(a,b,c,d)
        print(self.boardPieces[a - 1][b - 1].colour + " " + self.boardPieces[a - 1][b - 1].piece + " at " + str(a) + " " + str(b))
        print ("Computer Evaluation: " + str(val))
        print ("Number of Iterations: " + str(self.count))
        t1 = timer()
        print("Time elapsed: ", t1 - t0)

        self.count = 0
        self.isComputerMove = True
        
        # Set the piece to be moved and make it
        self.pieceToBeMoved.piece = self.boardPieces[a - 1][b - 1].piece
        self.pieceToBeMoved.colour = self.boardPieces[a - 1][b - 1].colour
        self.pieceToBeMoved.xLocation = self.boardPieces[a - 1][b - 1].xLocation
        self.pieceToBeMoved.yLocation = self.boardPieces[a - 1][b - 1].yLocation
        self.pieceToBeMoved.value = self.boardPieces[a - 1][b - 1].value
        
        self.pieceChosen = "Queen"
        
        self.move = True
        self.get_location(None, c, d)
        
        t1 = timer()
        print("Time elapsed: ", t1 - t0)
        print()
 

    else:
        
        print ("Position: " + str(val))
        print ("Number of Iterations: " + str(self.count))
        easygui.msgbox("Black Resigns", title="Winner!")
      
# Function to begin alpha beta decision making
def alphaBeta(self,curDepth,depthLimit):
      
    # Define the alpha and beta values
    alpha = -999999998
    beta = 999999999
    
    # If the full depth is reached, return the evaluation immediately
    if (curDepth >= depthLimit):
        return chess_eval.evaluate_board(pgnBoard)

    highestScore = -99999999
    # Acquire the prediction using the current board state and prepare variable for legal move filtering
    filteredPrediction = [0]*4096
    inputBoard = [encode_board(pgnBoard)]
    prediction = blackModel.predict(np.array(inputBoard),verbose=0)
        
    self.isComputerMove = True
    self.computerThinking = True
    
    x = 0
    y = 0
    w = 0
    z = 0
    
    # Create a pre-move copy of the board
    boardCopy = copy.deepcopy(self.boardPieces)

    # Filter the predictions to only contain legal moves
    for move in pgnBoard.legal_moves:
        cur = str(move)
        a = ord(cur[0:1]) - 96
        b = int(cur[1:2])
        c = ord(cur[2:3]) - 96
        d = int(cur[3:4])                     
        filteredPrediction[reversePrediction(a,b,c,d) - 1] = prediction[0][reversePrediction(a,b,c,d) - 1]
                        
    filteredPrediction = np.array(filteredPrediction)
    
    # Select the top moves to do a tree search
    for i in range (10):
       
        # Acquire current most likely move
        a,b,c,d = predictionInfo(np.argmax(filteredPrediction))
        # Zero out the current max so as to find the next max on the following turn
        filteredPrediction[np.argmax(filteredPrediction)] = 0
        if(reversePrediction(a,b,c,d) > 1):
            
           
            # Check if the move is a promotion and push the move accordingly
            if (self.boardPieces[c - 1][d - 1].piece == "Pawn" and b == 7 and d == 8):
                
                a = chr(a + 96)
                b = str(b)
                c = chr(c + 96)
                d = str(d)
                
                promotionPiece = 'q'
                pgnBoard.push(move.from_uci(a+b+c+d+promotionPiece))
                
            else:
                
                a = chr(a + 96)
                b = str(b)
                c = chr(c + 96)
                d = str(d)
                
                pgnBoard.push(move.from_uci(a+b+c+d))                
            
            # Increment the move number to simulate a move being made
            self.numMove += 1
            # Call the minimizer to make the next move
            score = minimizer(self, curDepth + 1,depthLimit,alpha, beta)
            self.numMove -= 1
            
            pgnBoard.pop()
            
            # Convert the characters back into coordinates
            a = ord(a) - 96
            b = int(b)
            c = ord(c) - 96
            d = int(d)

           
            # Find the highest score        
            if(score > highestScore):
                highestScore = score
                x = a
                y = b
                w = c
                z = d
            
            # Reset the board to the pre-moved state
            self.boardPieces = copy.deepcopy(boardCopy)
            
            alpha = max(alpha,highestScore)
        
            # If the beta value becomes less than the alpha value, the branch is not viable to find the best move
            if beta <= alpha:
                
                return x,y,w,z,highestScore     
                    
    if (curDepth == 0):
        
        return x,y,w,z,highestScore
    else:
        return highestScore        

# Function to find the maximum scoring move for a specific iteration
def maximizer(self,curDepth,depthLimit,alpha, beta):
    
    # If the full depth is reached, return the evaluation immediately
    if (curDepth >= depthLimit):
        return chess_eval.evaluate_board(pgnBoard)
    
    highestScore = -99999999   
    filteredPrediction = [0]*4096
    inputBoard = [encode_board(pgnBoard)]
    prediction = blackModel.predict(np.array(inputBoard),verbose=0)
    
    # Create a pre-move copy of the board
    boardCopy = copy.deepcopy(self.boardPieces)
    
    # Filter the predictions to only contain legal moves
    for move in pgnBoard.legal_moves:
        cur = str(move)
        a = ord(cur[0:1]) - 96
        b = int(cur[1:2])
        c = ord(cur[2:3]) - 96
        d = int(cur[3:4])                     
        filteredPrediction[reversePrediction(a,b,c,d) - 1] = prediction[0][reversePrediction(a,b,c,d) - 1]
                        
    filteredPrediction = np.array(filteredPrediction)
    
    # Select the top moves to do a tree search
    for i in range (10):
       
        # Acquire current most likely move
        a,b,c,d = predictionInfo(np.argmax(filteredPrediction))
        # Zero out the current max so as to find the next max on the following turn
        filteredPrediction[np.argmax(filteredPrediction)] = 0
        if(not(self.boardPieces[a-1][b-1] == None) and reversePrediction(a,b,c,d) > 1):
            
                       
            # Check if the move is a promotion and push the move accordingly
            if (self.boardPieces[c - 1][d - 1].piece == "Pawn" and b == 7 and d == 8):
                
                a = chr(a + 96)
                b = str(b)
                c = chr(c + 96)
                d = str(d)
                
                promotionPiece = 'q'
                pgnBoard.push(move.from_uci(a+b+c+d+promotionPiece))
                
            else:
                
                a = chr(a + 96)
                b = str(b)
                c = chr(c + 96)
                d = str(d)
                
                pgnBoard.push(move.from_uci(a+b+c+d))       
                            
            # Increment the move number to simulate a move being made
            self.numMove += 1
            # Call the minimizer to make the next move
            score = minimizer(self, curDepth + 1,depthLimit,alpha, beta)
            self.numMove -= 1

            # Undo the move to reset the board to pre-move state
            pgnBoard.pop()

            # Find the highest score
            if(score > highestScore):
                highestScore = score
                
            # Reset the board to the pre-moved state
            self.boardPieces = copy.deepcopy(boardCopy)
                
            alpha = max(alpha,highestScore)
        
            # If the beta value becomes less than the alpha value, the branch is not viable to find the best move
            if beta <= alpha:
                return highestScore                      
    return highestScore

# Function to find the minimum scoring move for a specific iteration      
def minimizer(self,curDepth,depthLimit,alpha, beta):
    
    # If the full depth is reached, return the evaluation immediately
    if (curDepth >= depthLimit):
        return chess_eval.evaluate_board(pgnBoard)
    
    lowestScore = 99999999
    filteredPrediction = [0]*4096
    inputBoard = [encode_board(pgnBoard)]
    prediction = whiteModel.predict(np.array(inputBoard),verbose=0)
    
    # Create a pre-move copy of the board
    boardCopy = copy.deepcopy(self.boardPieces)
    
    # Filter the predictions to only contain legal moves
    for move in pgnBoard.legal_moves:
        cur = str(move)
        a = ord(cur[0:1]) - 96
        b = int(cur[1:2])
        c = ord(cur[2:3]) - 96
        d = int(cur[3:4])                     
        filteredPrediction[reversePrediction(a,b,c,d) - 1] = prediction[0][reversePrediction(a,b,c,d) - 1]
                        
    filteredPrediction = np.array(filteredPrediction)
    
    # Select the top moves to do a tree search
    for i in range (15):
     
        # Acquire current most likely move
        a,b,c,d = predictionInfo(np.argmax(filteredPrediction))
        # Zero out the current max so as to find the next max on the following turn
        filteredPrediction[np.argmax(filteredPrediction)] = 0
        if(not(self.boardPieces[a-1][b-1] == None) and reversePrediction(a,b,c,d) > 1):
            
            # Check if the move is a promotion and push the move accordingly
            if (self.boardPieces[c - 1][d - 1].piece == "Pawn" and b == 7 and d == 8):
                
                a = chr(a + 96)
                b = str(b)
                c = chr(c + 96)
                d = str(d)
                
                promotionPiece = 'q'
                pgnBoard.push(move.from_uci(a+b+c+d+promotionPiece))
            else:
                
                a = chr(a + 96)
                b = str(b)
                c = chr(c + 96)
                d = str(d)
                
                pgnBoard.push(move.from_uci(a+b+c+d))        
            
            # Increment the move number to simulate a move being made
            self.numMove += 1
            # Call the maximizer to make the next move
            score = maximizer(self, curDepth + 1,depthLimit, alpha, beta)
            self.numMove -= 1
    
            # Undo the move to reset the board to premoved state
            pgnBoard.pop()        
    
            # Find the lowest score
            if(score < lowestScore):
                lowestScore = score
            
            # Reset the board to the pre-moved state
            self.boardPieces = copy.deepcopy(boardCopy)
                
            beta = min(beta,lowestScore)    
            
            # If the beta value becomes less than the alpha value, the branch is not viable to find the best move
            if beta <= alpha:
                return lowestScore  
    return lowestScore

def moveAppender(self,moves,y,colour):
    
    # All moves assume that white is on the bottom (Location numbers start at 1 but indices start at 0)
    if(colour =="Black"):
        oppositeColour = "White"
    else:
        oppositeColour = "Black"
    
    if (y.piece == "Pawn" ):
        
        if (colour == "Black"):
            if(not(y.yLocation == 2)):
                
                # Regular black pawn move - 1 down
                moves.append(Pieces(y.piece,y.colour,y.xLocation , y.yLocation - 1))
                
                # Black pawn capture to the right
                if (y.xLocation < 8):
                    if (self.boardPieces[y.xLocation][y.yLocation - 2].colour == oppositeColour):                          
                        moves.append(Pieces(y.piece,y.colour,y.xLocation + 1, y.yLocation - 1))
                    elif (self.whiteSideEnPasent and abs(y.xLocation - self.whiteSideEnPasentPawnxLocation) == 1 and y.yLocation == 4):
                        moves.append(Pieces(y.piece,y.colour,y.xLocation + 1, y.yLocation - 1))
                    
                # Black pawn capture to the left
                if (y.xLocation > 1):  
                    if (self.boardPieces[y.xLocation - 2][y.yLocation - 2].colour == oppositeColour):                          
                        moves.append(Pieces(y.piece,y.colour,y.xLocation - 1, y.yLocation - 1)) 
                    elif (self.whiteSideEnPasent and abs(y.xLocation - self.whiteSideEnPasentPawnxLocation) == 1 and y.yLocation == 5):
                        moves.append(Pieces(y.piece,y.colour,y.xLocation - 1, y.yLocation - 1))                          
                           
                # Black pawn moves 2 squares down initially
                if (y.yLocation == 7):                            
                    moves.append(Pieces(y.piece,y.colour,y.xLocation, y.yLocation - 2))
                    
            else:
                # Black pawn moves downward to promotion
                moves.append(Pieces("Queen",y.colour,y.xLocation , y.yLocation - 1))
                moves.append(Pieces("Knight",y.colour,y.xLocation , y.yLocation - 1)) 
                
                # Black pawn captures to the right to promotion
                if (y.xLocation < 8):                            
                    moves.append(Pieces("Queen",y.colour,y.xLocation + 1, y.yLocation - 1))
                    moves.append(Pieces("Knight",y.colour,y.xLocation + 1, y.yLocation - 1))
                
                # Black pawn captures to the left to promotion
                if (y.xLocation > 1):                            
                    moves.append(Pieces("Queen",y.colour,y.xLocation - 1, y.yLocation - 1))
                    moves.append(Pieces("Knight",y.colour,y.xLocation - 1, y.yLocation - 1))   
        else:
            if(not(y.yLocation == 7)):
                
                # Regular white pawn move - 1 up
                moves.append(Pieces(y.piece,y.colour,y.xLocation , y.yLocation + 1))

                # White pawn capture to the right
                if (y.xLocation < 8):
                    
                    if (self.boardPieces[y.xLocation][y.yLocation].colour == oppositeColour):                          
                        moves.append(Pieces(y.piece,y.colour,y.xLocation + 1, y.yLocation + 1))
                    elif (self.blackSideEnPasent and abs(y.xLocation - self.blackSideEnPasentPawnxLocation) == 1):
                        moves.append(Pieces(y.piece,y.colour,y.xLocation + 1, y.yLocation + 1))
                    
                # White pawn capture to the left
                if (y.xLocation > 1):

                    if (self.boardPieces[y.xLocation - 2][y.yLocation].colour == oppositeColour):                          
                        moves.append(Pieces(y.piece,y.colour,y.xLocation - 1, y.yLocation + 1)) 
                    elif (self.blackSideEnPasent and abs(y.xLocation - self.blackSideEnPasentPawnxLocation) == 1):
                        moves.append(Pieces(y.piece,y.colour,y.xLocation - 1, y.yLocation + 1)) 
                   
                # White pawn moves 2 squares up initially
                if (y.yLocation == 2):                            
                    moves.append(Pieces(y.piece,y.colour,y.xLocation, y.yLocation + 2))
            else:
                # White pawn moves upward to promotion
                moves.append(Pieces("Queen",y.colour,y.xLocation , y.yLocation + 1)) 
                moves.append(Pieces("Knight",y.colour,y.xLocation , y.yLocation + 1))
                
                # White pawn capture to the right to promotion
                if (y.xLocation < 8):                            
                    moves.append(Pieces("Queen",y.colour,y.xLocation + 1, y.yLocation + 1))
                    moves.append(Pieces("Knight",y.colour,y.xLocation + 1, y.yLocation + 1))
                    
                # White pawn capture to the left to promotion
                if (y.xLocation > 1):                            
                    moves.append(Pieces("Queen",y.colour,y.xLocation - 1, y.yLocation + 1)) 
                    moves.append(Pieces("Knight",y.colour,y.xLocation - 1, y.yLocation + 1)) 
        
    elif (y.piece == "Knight"):
        
        # Knight moves right 2 up 1
        if (y.xLocation + 2 < 9 and y.yLocation + 1 < 9 and not(self.boardPieces[y.xLocation + 1][y.yLocation].colour == colour)):
            moves.append(Pieces(y.piece,y.colour,y.xLocation + 2, y.yLocation + 1))
            
        # Knight moves right 2 down 1
        if (y.xLocation + 2 < 9 and y.yLocation - 1 > 0 and not(self.boardPieces[y.xLocation + 1][y.yLocation - 2].colour == colour)):
            moves.append(Pieces(y.piece,y.colour,y.xLocation + 2, y.yLocation - 1))
        
        # Knight moves left 2 up 1
        if (y.xLocation - 2 > 0 and y.yLocation + 1 < 9 and not(self.boardPieces[y.xLocation - 3][y.yLocation].colour == colour)):
            moves.append(Pieces(y.piece,y.colour,y.xLocation - 2, y.yLocation + 1))
        
        # Knight moves left 2 down 1
        if (y.xLocation - 2 > 0 and y.yLocation - 1 > 0 and not(self.boardPieces[y.xLocation - 3][y.yLocation - 2].colour == colour)):
            moves.append(Pieces(y.piece,y.colour,y.xLocation - 2, y.yLocation - 1))
        
        # Knight moves right 1 up 2
        if (y.xLocation + 1 < 9 and y.yLocation + 2 < 9 and not(self.boardPieces[y.xLocation][y.yLocation + 1].colour == colour)):
            moves.append(Pieces(y.piece,y.colour,y.xLocation + 1, y.yLocation + 2))
        
        # Knight moves right 1 down 2
        if (y.xLocation + 1 < 9 and y.yLocation - 2 > 0 and not(self.boardPieces[y.xLocation][y.yLocation - 3].colour == colour)):
            moves.append(Pieces(y.piece,y.colour,y.xLocation + 1, y.yLocation - 2))
        
        # Knight moves left 1 up 2    
        if (y.xLocation - 1 > 0 and y.yLocation + 2 < 9 and not(self.boardPieces[y.xLocation - 2][y.yLocation + 1].colour == colour)):
            moves.append(Pieces(y.piece,y.colour,y.xLocation - 1, y.yLocation + 2))
        
        # Knight moves left 1 down 2
        if (y.xLocation - 1 > 0 and y.yLocation - 2 > 0 and not(self.boardPieces[y.xLocation - 2][y.yLocation - 3].colour == colour)):
            moves.append(Pieces(y.piece,y.colour,y.xLocation - 1, y.yLocation - 2))
        
    elif (y.piece == "Bishop"):
        continuation = [True]* 4
        for i in range(1,8):
            
            # Bishop move upward right diagonal
            if(y.xLocation + i < 9 and y.yLocation + i < 9 and not(self.boardPieces[y.xLocation + i - 1][y.yLocation + i - 1].colour == colour) and continuation [0] == True):
                
                moves.append(Pieces(y.piece,y.colour,y.xLocation + i, y.yLocation + i))
                
                # Check if there is an obstacle preventing further exploration
                if(not(self.boardPieces[y.xLocation + i - 1][y.yLocation + i - 1].piece == "Empty")):
                    continuation [0] == False
            
            # Bishop move downward right diagonal       
            if(y.xLocation + i < 9 and y.yLocation - i > 0 and not(self.boardPieces[y.xLocation + i - 1][y.yLocation - i - 1].colour == colour) and continuation [1] == True):
                
                moves.append(Pieces(y.piece,y.colour,y.xLocation + i, y.yLocation - i))
                
                # Check if there is an obstacle preventing further exploration
                if(not(self.boardPieces[y.xLocation + i - 1][y.yLocation - i - 1].piece == "Empty")):
                    continuation [1] == False
            
            # Bishop move upward left diagonal
            if(y.xLocation - i > 0 and y.yLocation + i < 9 and not(self.boardPieces[y.xLocation - i - 1][y.yLocation + i - 1].colour == colour) and continuation [2] == True):
                
                moves.append(Pieces(y.piece,y.colour,y.xLocation - i, y.yLocation + i))
                
                # Check if there is an obstacle preventing further exploration
                if(not(self.boardPieces[y.xLocation - i - 1][y.yLocation + i - 1].piece == "Empty")):
                    continuation [2] == False
            
            # Bishop move downward left diagonal
            if(y.xLocation - i > 0 and y.yLocation - i > 0 and not(self.boardPieces[y.xLocation - i - 1][y.yLocation - i - 1].colour == colour) and continuation [3] == True):
                
                moves.append(Pieces(y.piece,y.colour,y.xLocation - i, y.yLocation - i))
                
                # Check if there is an obstacle preventing further exploration
                if(not(self.boardPieces[y.xLocation - i - 1][y.yLocation - i - 1].piece == "Empty")):
                    continuation [3] == False

        
    elif (y.piece == "Rook"):
        continuation = [True]* 4
        for i in range(1,8):
            
            # Rook move right
            if(y.xLocation + i < 9 and not(self.boardPieces[y.xLocation + i - 1][y.yLocation - 1].colour == colour) and continuation [0] == True):
                
                moves.append(Pieces(y.piece,y.colour,y.xLocation + i, y.yLocation))
                
                # Check if there is an obstacle preventing further exploration
                if(not(self.boardPieces[y.xLocation + i - 1][y.yLocation - 1].piece == "Empty")):
                    continuation [0] == False
                
            # Rook move down
            if(y.yLocation - i > 0 and not(self.boardPieces[y.xLocation - 1][y.yLocation - i - 1].colour == colour) and continuation [1] == True):
                
                moves.append(Pieces(y.piece,y.colour,y.xLocation, y.yLocation - i))
                
                # Check if there is an obstacle preventing further exploration
                if(not(self.boardPieces[y.xLocation - 1][y.yLocation - i - 1].piece == "Empty")):
                    continuation [1] == False
            
            # Rook move left
            if(y.xLocation - i > 0 and not(self.boardPieces[y.xLocation - i - 1][y.yLocation - 1].colour == colour) and continuation [2] == True):
                
                moves.append(Pieces(y.piece,y.colour,y.xLocation - i, y.yLocation))
                
                # Check if there is an obstacle preventing further exploration
                if(not(self.boardPieces[y.xLocation - i - 1][y.yLocation - 1].piece == "Empty")):
                    continuation [2] == False
            
            # Rook move up    
            if(y.yLocation + i < 9 and not(self.boardPieces[y.xLocation - 1][y.yLocation + i - 1].colour == colour) and continuation [3] == True):
                
                moves.append(Pieces(y.piece,y.colour,y.xLocation, y.yLocation + i))
                
                # Check if there is an obstacle preventing further exploration
                if(not(self.boardPieces[y.xLocation - 1][y.yLocation + i - 1].piece == "Empty")):
                    continuation [3] == False

        
    elif (y.piece == "Queen"):
        
        continuation = [True]* 8
        for i in range(1,8):
            
            # Diagonal Moves
            
            # Queen move upward right diagonal
            if(y.xLocation + i < 9 and y.yLocation + i < 9 and not(self.boardPieces[y.xLocation + i - 1][y.yLocation + i - 1].colour == colour) and continuation [0] == True):
                
                moves.append(Pieces(y.piece,y.colour,y.xLocation + i, y.yLocation + i))
                
                # Check if there is an obstacle preventing further exploration
                if(not(self.boardPieces[y.xLocation + i - 1][y.yLocation + i - 1].piece == "Empty")):
                    continuation [0] == False
            
            # Queen move downward right diagonal
            if(y.xLocation + i < 9 and y.yLocation - i > 0 and not(self.boardPieces[y.xLocation + i - 1][y.yLocation - i - 1].colour == colour) and continuation [1] == True):
                
                moves.append(Pieces(y.piece,y.colour,y.xLocation + i, y.yLocation - i))
                
                # Check if there is an obstacle preventing further exploration
                if(not(self.boardPieces[y.xLocation + i - 1][y.yLocation - i - 1].piece == "Empty")):
                    continuation [1] == False
            
            # Queen move upward left diagonal
            if(y.xLocation - i > 0 and y.yLocation + i < 9 and not(self.boardPieces[y.xLocation - i - 1][y.yLocation + i - 1].colour == colour) and continuation [2] == True):
                
                moves.append(Pieces(y.piece,y.colour,y.xLocation - i, y.yLocation + i))
                
                # Check if there is an obstacle preventing further exploration
                if(not(self.boardPieces[y.xLocation - i - 1][y.yLocation + i - 1].piece == "Empty")):
                    continuation [2] == False
            
            # Queen move downward left diagonal
            if(y.xLocation - i > 0 and y.yLocation - i > 0 and not(self.boardPieces[y.xLocation - i - 1][y.yLocation - i - 1].colour == colour) and continuation [3] == True):
                
                moves.append(Pieces(y.piece,y.colour,y.xLocation - i, y.yLocation - i))
                
                # Check if there is an obstacle preventing further exploration
                if(not(self.boardPieces[y.xLocation - i - 1][y.yLocation - i - 1].piece == "Empty")):
                    continuation [3] == False


            # Horizontal and Vertical Moves
            
            # Queen move right horizontally
            if(y.xLocation + i < 9 and not(self.boardPieces[y.xLocation + i - 1][y.yLocation - 1].colour == colour) and continuation [4] == True):
                
                moves.append(Pieces(y.piece,y.colour,y.xLocation + i, y.yLocation))
                
                # Check if there is an obstacle preventing further exploration
                if(not(self.boardPieces[y.xLocation + i - 1][y.yLocation - 1].piece == "Empty")):
                    continuation [4] == False
            
            # Queen move downward vertically
            if(y.yLocation - i > 0 and not(self.boardPieces[y.xLocation - 1][y.yLocation - i - 1].colour == colour) and continuation [5] == True):
                
                moves.append(Pieces(y.piece,y.colour,y.xLocation, y.yLocation - i))
                
                # Check if there is an obstacle preventing further exploration
                if(not(self.boardPieces[y.xLocation - 1][y.yLocation - i - 1].piece == "Empty")):
                    continuation [5] == False
            
            # Queen move left horizontally
            if(y.xLocation - i > 0 and not(self.boardPieces[y.xLocation - i - 1][y.yLocation - 1].colour == colour) and continuation [6] == True):
                
                moves.append(Pieces(y.piece,y.colour,y.xLocation - i, y.yLocation))
                
                # Check if there is an obstacle preventing further exploration
                if(not(self.boardPieces[y.xLocation - i - 1][y.yLocation - 1].piece == "Empty")):
                    continuation [6] == False
            
            # Queen move upward vertically
            if(y.yLocation + i < 9 and not(self.boardPieces[y.xLocation - 1][y.yLocation + i - 1].colour == colour) and continuation [7] == True):
                
                moves.append(Pieces(y.piece,y.colour,y.xLocation, y.yLocation + i))
                
                # Check if there is an obstacle preventing further exploration
                if(not(self.boardPieces[y.xLocation - 1][y.yLocation + i - 1].piece == "Empty")):
                    continuation [7] == False

        
    elif (y.piece == "King"):
 
        # Check if the King can move to the right
        if (y.xLocation + 1 < 9):
            moves.append(Pieces(y.piece,y.colour,y.xLocation + 1, y.yLocation))
            
            # Check if the king can also move up and to the right
            if(y.yLocation + 1 < 9):
                moves.append(Pieces(y.piece,y.colour,y.xLocation + 1, y.yLocation + 1))
            
            # Check if the king can also move down and to the right
            if(y.yLocation - 1 > 0):            
                moves.append(Pieces(y.piece,y.colour,y.xLocation + 1, y.yLocation - 1))
        
        # Check if the King can move to the left                    
        if(y.xLocation - 1 > 0):
            moves.append(Pieces(y.piece,y.colour,y.xLocation - 1, y.yLocation))
            
            # Check if the king can also move up and to the left
            if(y.yLocation + 1 < 9):
                moves.append(Pieces(y.piece,y.colour,y.xLocation - 1, y.yLocation + 1))
                
            # Check if the king can also move down and to the left
            if(y.yLocation - 1 > 0):
                moves.append(Pieces(y.piece,y.colour,y.xLocation - 1, y.yLocation - 1))
        
        # Check if the King can move up
        if (y.yLocation + 1 < 9):
            moves.append(Pieces(y.piece,y.colour,y.xLocation, y.yLocation + 1))
            
        # Check if the King can move down
        if (y.yLocation - 1 > 0):
            moves.append(Pieces(y.piece,y.colour,y.xLocation, y.yLocation - 1))
            
        # Check if White's King has moved for castling purposes    
        if(colour == "White" and not(self.whiteKingHasMoved)):
            
            # Check if the king side rook has moved
            if(not(self.kingSideWhiteRookHasMoved)):
                moves.append(Pieces(y.piece,y.colour,7, 1))
                
            # Check if the queen side rook has moved
            if(not(self.queenSideWhiteRookHasMoved)):
                moves.append(Pieces(y.piece,y.colour,3, 1))
                
        # Check if White's King has moved for castling purposes        
        elif(colour == "Black" and not(self.blackKingHasMoved)):
            
            # Check if the king side rook has moved
            if(not(self.kingSideBlackRookHasMoved)):
                moves.append(Pieces(y.piece,y.colour,7, 8))
                
            # Check if the queen side rook has moved
            if(not(self.queenSideBlackRookHasMoved)):
                moves.append(Pieces(y.piece,y.colour,3, 8))