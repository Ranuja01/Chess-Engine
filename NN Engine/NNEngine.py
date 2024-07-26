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
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, InputLayer
#from pickle import dump
from timeit import default_timer as timer
import chess
import chess.pgn
import io

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Set TensorFlow log level to suppress all but errors


pgnBoard = chess.Board()
pgnBoard.legal_moves

blackModel = tf.keras.models.load_model('../Models/BlackModel4.keras')
whiteModel = tf.keras.models.load_model('../Models/WhiteModel1.keras')

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
        if(Rules.isCheckMate(chess.WHITE)):
            print("AAAAA")
            sum = 10000000
        elif(Rules.isCheckMate(chess.BLACK)):
            print("BBBB")
            sum = -10000000
             
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
    currentItem,pieceToBeMoved,val,self.pieceChosen = alphaBeta(self, 0,self.depth,"Black")
    self.computerThinking = False
    
    # If the algorithm does not select a move, that suggests they are all equally bad and therefore the machine resigns
    if(not(pieceToBeMoved == None or pieceToBeMoved.piece == "Empty" or reversePrediction(self.pieceToBeMoved.xLocation,self.pieceToBeMoved.yLocation,currentItem.xLocation,currentItem.yLocation) == 1)):
        
        print(pieceToBeMoved.colour + " " + pieceToBeMoved.piece + " at " + str(pieceToBeMoved.xLocation) + " " + str(pieceToBeMoved.yLocation))
        print ("Computer Evaluation: " + str(val))
        print ("Number of Iterations: " + str(self.count))
        t1 = timer()
        print("Time elapsed: ", t1 - t0)

        self.count = 0
        self.isComputerMove = True
        
        # Set the piece to be moved and make it
        self.pieceToBeMoved.piece = pieceToBeMoved.piece
        self.pieceToBeMoved.colour = pieceToBeMoved.colour
        self.pieceToBeMoved.xLocation = pieceToBeMoved.xLocation
        self.pieceToBeMoved.yLocation = pieceToBeMoved.yLocation
        self.pieceToBeMoved.value = pieceToBeMoved.value
        
        # Convert the engine selected coordinates to the move string
        a = chr(self.pieceToBeMoved.xLocation + 96)
        b = str(self.pieceToBeMoved.yLocation)
        c = chr(currentItem.xLocation + 96)
        d = str(currentItem.yLocation)
        
        self.move = True
        self.get_location(None, currentItem.xLocation, currentItem.yLocation)
        
        t1 = timer()
        print("Time elapsed: ", t1 - t0)
        print()
 

    else:
        
        print ("Position: " + str(val))
        print ("Number of Iterations: " + str(self.count))
        easygui.msgbox("Black Resigns", title="Winner!")

        
# Function to begin alpha beta decision making
def alphaBeta(self,curDepth,depthLimit,evalColour):
      
    # Define the alpha and beta values
    alpha = -999999998
    beta = 999999999
    
    # Determine the colour of the active player and the opposing player
    if(evalColour == "Black"):
        oppositeColour = "White"
    else:
        oppositeColour = "Black"
    
    # If the full depth is reached, return the evaluation immediately
    if (curDepth >= depthLimit):
        return evaluateBoard(self)
    
    # Make copies of the en pasent status
    EnPasentCopy = self.whiteSideEnPasent
    EnPasentLocationCopy = self.whiteSideEnPasentPawnxLocation
    
    isLegal = False
    isCapture = False
    highestScore = -99999999
    pieceToBePromoted = "None"
    promotion = False
    curItem = Pieces ("Empty","None",0,0)
    pieceToBeMoved = Pieces ("Empty","None",0,0)
    
    item = Pieces ("Empty","None",0,0)
    castleFlag = False
    
    # Acquire the prediction using the current board state and prepare variable for legal move filtering
    filteredPrediction = [0]*4096
    inputBoard = [encode_board(pgnBoard)]
    prediction = blackModel.predict(np.array(inputBoard))
        
    self.isComputerMove = True
    self.computerThinking = True
    
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
       
        '''
        print("AAA", self.boardPieces[4][7].piece, self.boardPieces[4][7].colour)
          
        print("X1: ",a)
        print("Y1: ",b)
        print("X2: ",c)
        print("Y2: ",d)
        
        print(prediction[0][np.argmax(filteredPrediction)] * 100,"%")
        print()
        '''
        
        # Acquire current most likely move
        a,b,c,d = predictionInfo(np.argmax(filteredPrediction))
        # Zero out the current max so as to find the next max on the following turn
        filteredPrediction[np.argmax(filteredPrediction)] = 0
        if(not(self.boardPieces[a-1][b-1] == None) and reversePrediction(a,b,c,d) > 1):
            
            # Set the piece to be moved as the current piece                        
            self.pieceToBeMoved.piece = self.boardPieces[a - 1][b - 1].piece
            self.pieceToBeMoved.colour = self.boardPieces[a - 1][b - 1].colour
            self.pieceToBeMoved.value = self.boardPieces[a - 1][b - 1].value
            self.pieceToBeMoved.xLocation = a
            self.pieceToBeMoved.yLocation = b
            
            item.piece = self.boardPieces[a - 1][b - 1].piece
            item.colour = self.boardPieces[a - 1][b - 1].colour
            item.value = self.boardPieces[a - 1][b - 1].value
            item.xLocation = c
            item.yLocation = d
            
            # Check if moving this piece is a castling move
            # No need to check other legality since the moves are already filtered
            _, isLegal,isCapture = Rules.isLegalKingMove(self, self.boardPieces[item.xLocation - 1][item.yLocation - 1],item.xLocation,item.yLocation,evalColour)

            # Set the destination as the moving piece and the original location as empty
            self.boardPieces[item.xLocation - 1][item.yLocation - 1].piece = item.piece
            self.boardPieces[item.xLocation - 1][item.yLocation - 1].colour = item.colour
            self.boardPieces[item.xLocation - 1][item.yLocation - 1].value = item.value
            
            self.boardPieces[a - 1][b - 1].piece = "Empty"
            self.boardPieces[a - 1][b - 1].colour = "None"
            self.boardPieces[a - 1][b - 1].value = 0
            
            # Check if the move made is a castling move
            if(self.isCastle):
                
                # Set the castle flag within the scope of the current move
                castleFlag = True
                if(evalColour == "Black"):
                    
                    self.blackHasCastled = True
                    if(item.xLocation == 7):
                        
                        # Move the rook to complete kingside castling for black
                        self.boardPieces[5][7].piece = self.boardPieces[7][7].piece
                        self.boardPieces[5][7].colour = evalColour
                        self.boardPieces[5][7].value = self.boardPieces[7][7].value
                        
                        self.boardPieces[7][7].piece = "Empty"
                        self.boardPieces[7][7].colour = "None"
                        self.boardPieces[7][7].value = 0
                    
                    elif(item.xLocation == 3):
                        
                        # Move the rook to complete queenside castling for black
                        self.boardPieces[3][7].piece = self.boardPieces[0][7].piece
                        self.boardPieces[3][7].colour = evalColour
                        self.boardPieces[3][7].value = self.boardPieces[0][7].value
                        
                        self.boardPieces[0][7].piece = "Empty"
                        self.boardPieces[0][7].colour = "None"
                        self.boardPieces[0][7].value = 0
                    
                else:
                    
                    self.whiteHasCastled = True
                    if(item.xLocation == 7):
                        
                        # Move the rook to complete kingside castling for white
                        self.boardPieces[5][0].piece = self.boardPieces[7][0].piece
                        self.boardPieces[5][0].colour = evalColour
                        self.boardPieces[5][0].value = self.boardPieces[7][0].value
                        
                        self.boardPieces[7][0].piece = "Empty"
                        self.boardPieces[7][0].colour = "None"
                        self.boardPieces[7][0].value = 0
                    
                    elif(item.xLocation == 3):
                        
                        # Move the rook to complete queenside castling for white
                        self.boardPieces[3][0].piece = self.boardPieces[0][0].piece
                        self.boardPieces[3][0].colour = evalColour
                        self.boardPieces[3][0].value = self.boardPieces[0][0].value
                        
                        self.boardPieces[0][0].piece = "Empty"
                        self.boardPieces[0][0].colour = "None"
                        self.boardPieces[0][0].value = 0
               
                # Set the universal castle flag as false to return to the previous state
                self.isCastle = False       
            
            if (evalColour == "Black"):
                
                # If the piece was set in a previous iteration as in en pasent position, it is no longer by the next move
                if(self.blackSideEnPasent):
                    self.blackSideEnPasent = False
                
                # Set if the pawn is in en pasent position
                if(self.pieceToBeMoved.piece == "Pawn" and item.yLocation == 5 and b == 7):
                    self.blackSideEnPasent = True
                    self.blackSideEnPasentPawnxLocation = a
                    pass
                
                # Remove the pawn being captured if done through en pasent
                if(self.whiteSideEnPasent and self.pieceToBeMoved.piece == "Pawn" and item.xLocation == self.whiteSideEnPasentPawnxLocation and isCapture and b == 4):
                    self.boardPieces[self.whiteSideEnPasentPawnxLocation - 1][3].piece = "Empty"
                    self.boardPieces[self.whiteSideEnPasentPawnxLocation - 1][3].colour = "None"
                    self.boardPieces[self.whiteSideEnPasentPawnxLocation - 1][3].value = 0
                                    
            else:
                
                # If the piece was set in a previous iteration as in en pasent position, it is no longer by the next move
                if(self.whiteSideEnPasent):
                    self.whiteSideEnPasent = False
                
                # Set if the pawn is in en pasent position
                if(self.pieceToBeMoved.piece == "Pawn" and item.yLocation == 4 and b == 2):
                    self.whiteSideEnPasent = True
                    self.whiteSideEnPasentPawnxLocation = a  
                    pass 
                
                # Remove the pawn being captured if done through en pasent
                if(self.blackSideEnPasent and self.pieceToBeMoved.piece == "Pawn" and item.xLocation == self.blackSideEnPasentPawnxLocation and isCapture and b == 5):
                    self.boardPieces[self.blackSideEnPasentPawnxLocation - 1][4].piece = "Empty"
                    self.boardPieces[self.blackSideEnPasentPawnxLocation - 1][4].colour = "None"
                    self.boardPieces[self.blackSideEnPasentPawnxLocation - 1][4].value = 0
           
            # Check if the move is a promotion and push the move accordingly
            if (self.boardPieces[c - 1][d - 1].piece == "Pawn" and b == 7 and d == 8):
                
                a = chr(a + 96)
                b = str(b)
                c = chr(c + 96)
                d = str(d)
                
                promotionPiece = 'q'
                pgnBoard.push(move.from_uci(a+b+c+d+promotionPiece))
                self.boardPieces[item.xLocation - 1][item.yLocation - 1].piece = "Queen"
                self.boardPieces[item.xLocation - 1][item.yLocation - 1].value = 9000
            else:
                
                a = chr(a + 96)
                b = str(b)
                c = chr(c + 96)
                d = str(d)
                
                pgnBoard.push(move.from_uci(a+b+c+d))                
            
            # Increment the move number to simulate a move being made
            self.numMove += 1
            # Call the minimizer to make the next move
            score = minimizer(self, curDepth + 1,depthLimit,oppositeColour,alpha, beta)
            self.numMove -= 1
            
            pgnBoard.pop()
            
            self.whiteSideEnPasent = EnPasentCopy
            self.whiteSideEnPasentPawnxLocation = EnPasentLocationCopy

            # Convert the characters back into coordinates
            a = ord(a) - 96
            b = int(b)
            c = ord(c) - 96
            d = int(d)

            # If the in-scope castling flag was set, then reset the universal castle flags
            if (castleFlag):
                
                castleFlag = False
                if (evalColour == "Black"):
                    self.blackHasCastled = False
                else:
                    self.whiteHasCastled = False                                     
                    
            # Find the highest score        
            if(score > highestScore):
                
                # a pawn is to be promoted, set the promotion piece as the chosen one
                if(promotion):
                    pieceToBePromoted = item.piece
                highestScore = score
                
                # Set the destination location
                curItem.xLocation = item.xLocation 
                curItem.yLocation = item.yLocation
                
                # Reset the board to the pre-moved state
                self.boardPieces = copy.deepcopy(boardCopy)
                
                # Set the piece to be moved
                pieceToBeMoved.piece = self.boardPieces[a - 1][b - 1].piece
                pieceToBeMoved.colour = self.boardPieces[a - 1][b - 1].colour
                pieceToBeMoved.xLocation = a
                pieceToBeMoved.yLocation = b
                pieceToBeMoved.value = self.boardPieces[a - 1][b - 1].value
            
            # Reset the promotion variable
            promotion = False
            # Reset the board to the pre-moved state
            self.boardPieces = copy.deepcopy(boardCopy)
            
            alpha = max(alpha,highestScore)
        
            # If the beta value becomes less than the alpha value, the branch is not viable to find the best move
            if beta <= alpha:
                self.whiteSideEnPasent = EnPasentCopy
                self.whiteSideEnPasentPawnxLocation = EnPasentLocationCopy
                return curItem,pieceToBeMoved,highestScore,pieceToBePromoted     
                    
    if (curDepth == 0):
        self.whiteSideEnPasent = EnPasentCopy
        self.whiteSideEnPasentPawnxLocation = EnPasentLocationCopy
        return curItem,pieceToBeMoved,highestScore,pieceToBePromoted
    else:
        return highestScore        


# Function to find the maximum scoring move for a specific iteration
def maximizer(self,curDepth,depthLimit,evalColour,alpha, beta):
    
    # Determine the colour of the active player and the opposing player
    if(evalColour == "Black"):
        oppositeColour = "White"
    else:
        oppositeColour = "Black"
    
    # If the full depth is reached, return the evaluation immediately
    if (curDepth >= depthLimit):
        return evaluateBoard(self)
    
    isLegal = False
    isCapture = False
    highestScore = -99999999
    castleFlag = False
    item = Pieces ("Empty","None",0,0)
    
    filteredPrediction = [0]*4096
   
    inputBoard = [encode_board(pgnBoard)]
    #activeModel = keras.models.Sequential()
    prediction = blackModel.predict(np.array(inputBoard))
    
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
    for i in range (5):
       
        '''
        print("AAA", self.boardPieces[4][7].piece, self.boardPieces[4][7].colour)
          
        print("X1: ",a)
        print("Y1: ",b)
        print("X2: ",c)
        print("Y2: ",d)
        
        print(prediction[0][np.argmax(filteredPrediction)] * 100,"%")
        print()
        '''
        
        # Acquire current most likely move
        a,b,c,d = predictionInfo(np.argmax(filteredPrediction))
        # Zero out the current max so as to find the next max on the following turn
        filteredPrediction[np.argmax(filteredPrediction)] = 0
        if(not(self.boardPieces[a-1][b-1] == None) and reversePrediction(a,b,c,d) > 1):
            
            # Set the piece to be moved as the current piece                        
            self.pieceToBeMoved.piece = self.boardPieces[a - 1][b - 1].piece
            self.pieceToBeMoved.colour = self.boardPieces[a - 1][b - 1].colour
            self.pieceToBeMoved.value = self.boardPieces[a - 1][b - 1].value
            self.pieceToBeMoved.xLocation = a
            self.pieceToBeMoved.yLocation = b
            
            item.piece = self.boardPieces[a - 1][b - 1].piece
            item.colour = self.boardPieces[a - 1][b - 1].colour
            item.value = self.boardPieces[a - 1][b - 1].value
            item.xLocation = c
            item.yLocation = d
            
            # Check if moving this piece is a castling move
            # No need to check other legality since the moves are already filtered
            _, isLegal,isCapture = Rules.isLegalKingMove(self, self.boardPieces[item.xLocation - 1][item.yLocation - 1],item.xLocation,item.yLocation,evalColour)

            # Set the destination as the moving piece and the original location as empty
            self.boardPieces[item.xLocation - 1][item.yLocation - 1].piece = item.piece
            self.boardPieces[item.xLocation - 1][item.yLocation - 1].colour = item.colour
            self.boardPieces[item.xLocation - 1][item.yLocation - 1].value = item.value
            
            self.boardPieces[a - 1][b - 1].piece = "Empty"
            self.boardPieces[a - 1][b - 1].colour = "None"
            self.boardPieces[a - 1][b - 1].value = 0
            
            # Check if the move made is a castling move
            if(self.isCastle):
                
                # Set the castle flag within the scope of the current move
                castleFlag = True
                if(evalColour == "Black"):
                    
                    self.blackHasCastled = True
                    if(item.xLocation == 7):
                        
                        # Move the rook to complete kingside castling for black
                        self.boardPieces[5][7].piece = self.boardPieces[7][7].piece
                        self.boardPieces[5][7].colour = evalColour
                        self.boardPieces[5][7].value = self.boardPieces[7][7].value
                        
                        self.boardPieces[7][7].piece = "Empty"
                        self.boardPieces[7][7].colour = "None"
                        self.boardPieces[7][7].value = 0
                    
                    elif(item.xLocation == 3):
                        
                        # Move the rook to complete queenside castling for black
                        self.boardPieces[3][7].piece = self.boardPieces[0][7].piece
                        self.boardPieces[3][7].colour = evalColour
                        self.boardPieces[3][7].value = self.boardPieces[0][7].value
                        
                        self.boardPieces[0][7].piece = "Empty"
                        self.boardPieces[0][7].colour = "None"
                        self.boardPieces[0][7].value = 0
                    
                else:
                    
                    self.whiteHasCastled = True
                    if(item.xLocation == 7):
                        
                        # Move the rook to complete kingside castling for white
                        self.boardPieces[5][0].piece = self.boardPieces[7][0].piece
                        self.boardPieces[5][0].colour = evalColour
                        self.boardPieces[5][0].value = self.boardPieces[7][0].value
                        
                        self.boardPieces[7][0].piece = "Empty"
                        self.boardPieces[7][0].colour = "None"
                        self.boardPieces[7][0].value = 0
                    
                    elif(item.xLocation == 3):
                        
                        # Move the rook to complete queenside castling for white
                        self.boardPieces[3][0].piece = self.boardPieces[0][0].piece
                        self.boardPieces[3][0].colour = evalColour
                        self.boardPieces[3][0].value = self.boardPieces[0][0].value
                        
                        self.boardPieces[0][0].piece = "Empty"
                        self.boardPieces[0][0].colour = "None"
                        self.boardPieces[0][0].value = 0
               
                # Set the universal castle flag as false to return to the previous state
                self.isCastle = False       
            
            if (evalColour == "Black"):
                
                # If the piece was set in a previous iteration as in en pasent position, it is no longer by the next move
                if(self.blackSideEnPasent):
                    self.blackSideEnPasent = False
                
                # Set if the pawn is in en pasent position
                if(self.pieceToBeMoved.piece == "Pawn" and item.yLocation == 5 and b == 7):
                    self.blackSideEnPasent = True
                    self.blackSideEnPasentPawnxLocation = a
                    pass
                
                # Remove the pawn being captured if done through en pasent
                if(self.whiteSideEnPasent and self.pieceToBeMoved.piece == "Pawn" and item.xLocation == self.whiteSideEnPasentPawnxLocation and isCapture and b == 4):
                    self.boardPieces[self.whiteSideEnPasentPawnxLocation - 1][3].piece = "Empty"
                    self.boardPieces[self.whiteSideEnPasentPawnxLocation - 1][3].colour = "None"
                    self.boardPieces[self.whiteSideEnPasentPawnxLocation - 1][3].value = 0
                                    
            else:
                
                # If the piece was set in a previous iteration as in en pasent position, it is no longer by the next move
                if(self.whiteSideEnPasent):
                    self.whiteSideEnPasent = False
                
                # Set if the pawn is in en pasent position
                if(self.pieceToBeMoved.piece == "Pawn" and item.yLocation == 4 and b == 2):
                    self.whiteSideEnPasent = True
                    self.whiteSideEnPasentPawnxLocation = a  
                    pass 
                
                # Remove the pawn being captured if done through en pasent
                if(self.blackSideEnPasent and self.pieceToBeMoved.piece == "Pawn" and item.xLocation == self.blackSideEnPasentPawnxLocation and isCapture and b == 5):
                    self.boardPieces[self.blackSideEnPasentPawnxLocation - 1][4].piece = "Empty"
                    self.boardPieces[self.blackSideEnPasentPawnxLocation - 1][4].colour = "None"
                    self.boardPieces[self.blackSideEnPasentPawnxLocation - 1][4].value = 0
           
            # Check if the move is a promotion and push the move accordingly
            if (self.boardPieces[c - 1][d - 1].piece == "Pawn" and b == 7 and d == 8):
                
                a = chr(a + 96)
                b = str(b)
                c = chr(c + 96)
                d = str(d)
                
                promotionPiece = 'q'
                pgnBoard.push(move.from_uci(a+b+c+d+promotionPiece))
                self.boardPieces[item.xLocation - 1][item.yLocation - 1].piece = "Queen"
                self.boardPieces[item.xLocation - 1][item.yLocation - 1].value = 9000
            else:
                
                a = chr(a + 96)
                b = str(b)
                c = chr(c + 96)
                d = str(d)
                
                pgnBoard.push(move.from_uci(a+b+c+d))       
                            
            # Increment the move number to simulate a move being made
            self.numMove += 1
            # Call the minimizer to make the next move
            score = minimizer(self, curDepth + 1,depthLimit,oppositeColour,alpha, beta)
            self.numMove -= 1

            # Undo the move to reset the board to pre-move state
            pgnBoard.pop()

            # If the in-scope castling flag was set, then reset the universal castle flags
            if (castleFlag):
                castleFlag = False
                if (evalColour == "Black"):
                    self.blackHasCastled = False
                else:
                    self.whiteHasCastled = False 
            
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
def minimizer(self,curDepth,depthLimit,evalColour,alpha, beta):
    
    # Determine the colour of the active player and the opposing player
    if(evalColour == "Black"):
        oppositeColour = "White"
    else:
        oppositeColour = "Black"
    
    # If the full depth is reached, return the evaluation immediately
    if (curDepth >= depthLimit):
        return evaluateBoard(self)
    
    isLegal = False
    isCapture = False
    lowestScore = 99999999
    castleFlag = False
    item = Pieces ("Empty","None",0,0)
    
    filteredPrediction = [0]*4096
   
    inputBoard = [encode_board(pgnBoard)]
    #activeModel = keras.models.Sequential()
    prediction = whiteModel.predict(np.array(inputBoard))
    
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
    for i in range (5):
       
        '''
        print("AAA", self.boardPieces[4][7].piece, self.boardPieces[4][7].colour)
          
        print("X1: ",a)
        print("Y1: ",b)
        print("X2: ",c)
        print("Y2: ",d)
        
        print(prediction[0][np.argmax(filteredPrediction)] * 100,"%")
        print()
        '''
        
        # Acquire current most likely move
        a,b,c,d = predictionInfo(np.argmax(filteredPrediction))
        # Zero out the current max so as to find the next max on the following turn
        filteredPrediction[np.argmax(filteredPrediction)] = 0
        if(not(self.boardPieces[a-1][b-1] == None) and reversePrediction(a,b,c,d) > 1):
            
            # Set the piece to be moved as the current piece                        
            self.pieceToBeMoved.piece = self.boardPieces[a - 1][b - 1].piece
            self.pieceToBeMoved.colour = self.boardPieces[a - 1][b - 1].colour
            self.pieceToBeMoved.value = self.boardPieces[a - 1][b - 1].value
            self.pieceToBeMoved.xLocation = a
            self.pieceToBeMoved.yLocation = b
            
            item.piece = self.boardPieces[a - 1][b - 1].piece
            item.colour = self.boardPieces[a - 1][b - 1].colour
            item.value = self.boardPieces[a - 1][b - 1].value
            item.xLocation = c
            item.yLocation = d
            
            # Check if moving this piece is a castling move
            # No need to check other legality since the moves are already filtered
            _, isLegal,isCapture = Rules.isLegalKingMove(self, self.boardPieces[item.xLocation - 1][item.yLocation - 1],item.xLocation,item.yLocation,evalColour)

            # Set the destination as the moving piece and the original location as empty
            self.boardPieces[item.xLocation - 1][item.yLocation - 1].piece = item.piece
            self.boardPieces[item.xLocation - 1][item.yLocation - 1].colour = item.colour
            self.boardPieces[item.xLocation - 1][item.yLocation - 1].value = item.value
            
            self.boardPieces[a - 1][b - 1].piece = "Empty"
            self.boardPieces[a - 1][b - 1].colour = "None"
            self.boardPieces[a - 1][b - 1].value = 0
            
            # Check if the move made is a castling move
            if(self.isCastle):
                
                # Set the castle flag within the scope of the current move
                castleFlag = True
                if(evalColour == "Black"):
                    
                    self.blackHasCastled = True
                    if(item.xLocation == 7):
                        
                        # Move the rook to complete kingside castling for black
                        self.boardPieces[5][7].piece = self.boardPieces[7][7].piece
                        self.boardPieces[5][7].colour = evalColour
                        self.boardPieces[5][7].value = self.boardPieces[7][7].value
                        
                        self.boardPieces[7][7].piece = "Empty"
                        self.boardPieces[7][7].colour = "None"
                        self.boardPieces[7][7].value = 0
                    
                    elif(item.xLocation == 3):
                        
                        # Move the rook to complete queenside castling for black
                        self.boardPieces[3][7].piece = self.boardPieces[0][7].piece
                        self.boardPieces[3][7].colour = evalColour
                        self.boardPieces[3][7].value = self.boardPieces[0][7].value
                        
                        self.boardPieces[0][7].piece = "Empty"
                        self.boardPieces[0][7].colour = "None"
                        self.boardPieces[0][7].value = 0
                    
                else:
                    
                    self.whiteHasCastled = True
                    if(item.xLocation == 7):
                        
                        # Move the rook to complete kingside castling for white
                        self.boardPieces[5][0].piece = self.boardPieces[7][0].piece
                        self.boardPieces[5][0].colour = evalColour
                        self.boardPieces[5][0].value = self.boardPieces[7][0].value
                        
                        self.boardPieces[7][0].piece = "Empty"
                        self.boardPieces[7][0].colour = "None"
                        self.boardPieces[7][0].value = 0
                    
                    elif(item.xLocation == 3):
                        
                        # Move the rook to complete queenside castling for white
                        self.boardPieces[3][0].piece = self.boardPieces[0][0].piece
                        self.boardPieces[3][0].colour = evalColour
                        self.boardPieces[3][0].value = self.boardPieces[0][0].value
                        
                        self.boardPieces[0][0].piece = "Empty"
                        self.boardPieces[0][0].colour = "None"
                        self.boardPieces[0][0].value = 0
               
                # Set the universal castle flag as false to return to the previous state
                self.isCastle = False       
            
            if (evalColour == "Black"):
                
                # If the piece was set in a previous iteration as in en pasent position, it is no longer by the next move
                if(self.blackSideEnPasent):
                    self.blackSideEnPasent = False
                
                # Set if the pawn is in en pasent position
                if(self.pieceToBeMoved.piece == "Pawn" and item.yLocation == 5 and b == 7):
                    self.blackSideEnPasent = True
                    self.blackSideEnPasentPawnxLocation = a
                    pass
                
                # Remove the pawn being captured if done through en pasent
                if(self.whiteSideEnPasent and self.pieceToBeMoved.piece == "Pawn" and item.xLocation == self.whiteSideEnPasentPawnxLocation and isCapture and b == 4):
                    self.boardPieces[self.whiteSideEnPasentPawnxLocation - 1][3].piece = "Empty"
                    self.boardPieces[self.whiteSideEnPasentPawnxLocation - 1][3].colour = "None"
                    self.boardPieces[self.whiteSideEnPasentPawnxLocation - 1][3].value = 0
                                    
            else:
                
                # If the piece was set in a previous iteration as in en pasent position, it is no longer by the next move
                if(self.whiteSideEnPasent):
                    self.whiteSideEnPasent = False
                
                # Set if the pawn is in en pasent position
                if(self.pieceToBeMoved.piece == "Pawn" and item.yLocation == 4 and b == 2):
                    self.whiteSideEnPasent = True
                    self.whiteSideEnPasentPawnxLocation = a  
                    pass 
                
                # Remove the pawn being captured if done through en pasent
                if(self.blackSideEnPasent and self.pieceToBeMoved.piece == "Pawn" and item.xLocation == self.blackSideEnPasentPawnxLocation and isCapture and b == 5):
                    self.boardPieces[self.blackSideEnPasentPawnxLocation - 1][4].piece = "Empty"
                    self.boardPieces[self.blackSideEnPasentPawnxLocation - 1][4].colour = "None"
                    self.boardPieces[self.blackSideEnPasentPawnxLocation - 1][4].value = 0
           
            # Check if the move is a promotion and push the move accordingly
            if (self.boardPieces[c - 1][d - 1].piece == "Pawn" and b == 7 and d == 8):
                
                a = chr(a + 96)
                b = str(b)
                c = chr(c + 96)
                d = str(d)
                
                promotionPiece = 'q'
                pgnBoard.push(move.from_uci(a+b+c+d+promotionPiece))
                self.boardPieces[item.xLocation - 1][item.yLocation - 1].piece = "Queen"
                self.boardPieces[item.xLocation - 1][item.yLocation - 1].value = 9000
            else:
                
                a = chr(a + 96)
                b = str(b)
                c = chr(c + 96)
                d = str(d)
                
                pgnBoard.push(move.from_uci(a+b+c+d))        
            
            # Increment the move number to simulate a move being made
            self.numMove += 1
            # Call the maximizer to make the next move
            score = maximizer(self, curDepth + 1,depthLimit,oppositeColour, alpha, beta)
            self.numMove -= 1
    
            # Undo the move to reset the board to premoved state
            pgnBoard.pop()        
    
            # If the in-scope castling flag was set, then reset the universal castle flags
            if (castleFlag):
                castleFlag = False
                if (evalColour == "Black"):
                    self.blackHasCastled = False
                else:
                    self.whiteHasCastled = False  
            
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
    #print("AAA", self.boardPieces[1][6].piece, self.boardPieces[1][6].colour)
    #print("BBB", self.boardPieces[2][7].piece, self.boardPieces[2][7].colour)
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