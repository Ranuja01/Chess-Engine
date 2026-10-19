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

pgnBoard = chess.Board()
pgnBoard.legal_moves

model = tf.keras.models.load_model('../Models/BlackModel4.keras')

newPgn = io.StringIO("1. e4*")
newGame = chess.pgn.read_game(newPgn)

for move in newGame.mainline_moves():
    pass

def evaluateBoard(self):
    
    layer = [[
            [0,0,0,0,0,0,0,0], # First Column
            [0,0,5,5,5,5,3,0], # Second Column
            [0,0,3,15,25,5,0,0], # Third Column
            [0,0,3,25,30,15,0,0], # Fourth Column
            [0,0,3,25,30,15,0,0], # Fifth Column
            [0,0,3,15,25,5,0,0], # Sixth Column
            [0,0,5,5,5,5,3,0], # Seventh Column
            [0,0,0,0,0,0,0,0] # Eigth Column
            ],
            [
            [0,0,0,0,0,0,0,0], # First Column
            [0,0,3,5,5,5,0,0], # Second Column
            [0,0,5,25,15,3,0,0], # Third Column
            [0,0,15,30,25,3,0,0], # Fourth Column
            [0,0,15,30,25,3,0,0], # Fifth Column
            [0,0,5,25,15,3,0,0], # Sixth Column
            [0,0,3,5,5,5,0,0], # Seventh Column
            [0,0,0,0,0,0,0,0] # Eigth Column
            ]]
    
    placementLayer = [[
            [0,0,0,0,0,0,0,0], # First Column
            [-150,0,2,5,5,2,0,0], # Second Column
            [-200,0,3,15,15,5,0,0], # Third Column
            [0,-200,3,25,70,5,0,0], # Fourth Column
            [0,-200,3,25,70,5,0,0], # Fifth Column
            [-200,0,3,15,15,5,0,0], # Sixth Column
            [-150,0,2,5,5,2,0,0], # Seventh Column
            [0,0,0,0,0,0,0,0] # Eigth Column
            ],
            [
            [0,0,0,0,0,0,0,0], # First Column
            [0,0,2,5,5,2,0,-150], # Second Column
            [0,0,5,15,15,3,0,-200], # Third Column
            [0,0,5,70,25,3,-200,0], # Fourth Column
            [0,0,5,70,25,3,-200,0], # Fifth Column
            [0,0,5,15,15,3,0,-200], # Sixth Column
            [0,0,2,5,5,2,0,-150], # Seventh Column
            [0,0,0,0,0,0,0,0] # Eigth Column
            ]]
        
    
    layer2 = [[
             [0,0,1,2,3,15,40,0], # First Column 
             [0,0,2,5,15,25,40,0], # Second Column
             [0,0,3,5,25,35,40,0], # Third Column
             [0,0,3,5,25,35,40,0], # Fourth Column
             [0,0,3,5,25,35,40,0], # Fifth Column
             [0,0,3,5,25,35,40,0], # Sixth Column
             [0,0,2,5,15,25,40,0], # Seventh Column
             [0,0,1,2,3,15,40,0] # Eigth Column
             ],
             [
             [0,40,15,3,2,1,0,0], # First Column
             [0,40,25,15,5,2,0,0], # Second Column
             [0,40,35,25,5,3,0,0], # Third Column
             [0,40,35,25,5,3,0,0], # Fourth Column
             [0,40,35,25,5,3,0,0], # Fifth Column
             [0,40,35,25,5,3,0,0], # Sixth Column
             [0,40,25,15,5,2,0,0], # Seventh Column
             [0,40,15,3,2,1,0,0]    
                
             ]]
    
    if(self.move >= 18):
        activeLayer = layer2
        activePlacementLayer = layer2
    else:
        activeLayer = layer
        activePlacementLayer = placementLayer
    
    sum = 0

    #print ('\n')
    if(self.blackHasCastled):
        sum += 1500
    if(self.whiteHasCastled):
        sum -= 1500

    if(Rules.isInCheck(self, "White")):
        if(Rules.isCheckMate(self, "White")):
            #print("AAAAA")
            sum = 10000000
    elif(Rules.isInCheck(self, "Black")):
        if(Rules.isCheckMate(self, "Black")):
            #print("BBBB")
            sum = -10000000
            
    
    # Check for center square control
    for item in self.boardPieces:
        for square in item:
            if(not(square.piece == "King")):
                
                if (square.colour == "Black"):
                    sum += square.value
                    #if(square.piece == "Pawn"):
                        
                    if(not(square.piece == "Rook" and activePlacementLayer == placementLayer)):
                        sum += activePlacementLayer[1][square.xLocation - 1][square.yLocation - 1]
                elif (square.colour == "White"):
                    sum -= square.value
                    #if(square.piece == "Pawn"):
                        
                    if(not(square.piece == "Rook" and activePlacementLayer == placementLayer)):
                        sum -= activePlacementLayer[0][square.xLocation - 1][square.yLocation - 1]
                        
                if(sum >= 13200):
                    activeLayer = layer
                    activePlacementLayer = placementLayer
                elif(self.numMove >= 18):
                    if(square.piece == "Pawn"):
                        if(square.colour == "Black"):
                            sum += (9 - square.yLocation) * 25
                        else:
                            sum -= square.yLocation * 25
    
                if(square.xLocation == 4 and square.yLocation == 4 or square.xLocation == 4 and square.yLocation == 5 or square.xLocation == 5 and square.yLocation == 4 or square.xLocation == 5 and square.yLocation == 5):
                    #print("AAAAA", square.piece,square.colour,square.xLocation,square.yLocation, sum)
                    if (square.colour == "Black"):
                        sum += 150
                    elif (square.colour == "White"):
                        sum -= 150                       
    return sum     

def redesignBoard(board):
    newBoard = [[]]
    for i in range(112,127,2):
        for j in range(0,8):
            newBoard[0].append(board[i - j * 16])
    return newBoard

def predictionInfo(prediction):
    #print("ASDASDASD")
    pieceToBeMoved = (prediction // 64) + 1
    #print(pieceToBeMoved)    
    pieceToBeMovedXLocation = (pieceToBeMoved // 8) + 1    
    pieceToBeMovedYLocation = pieceToBeMoved % 8
    
    remainder = prediction % 64
    squareToBeMovedToXLocation = (remainder // 8) + 1
    squareToBeMovedToYLocation = remainder % 8 
    
    if (squareToBeMovedToYLocation == 0):
        squareToBeMovedToYLocation = 8
        squareToBeMovedToXLocation -= 1
        
    if (pieceToBeMovedYLocation == 0):
        pieceToBeMovedYLocation = 8
        pieceToBeMovedXLocation -= 1

    if(squareToBeMovedToXLocation == 0):
        
        squareToBeMovedToXLocation = 8
        pieceToBeMovedYLocation -= 1
    
    if (squareToBeMovedToYLocation == 0):
        squareToBeMovedToYLocation = 8
        squareToBeMovedToXLocation -= 1
        
    if (pieceToBeMovedYLocation == 0):
        pieceToBeMovedYLocation = 8
        pieceToBeMovedXLocation -= 1
    
    
    return pieceToBeMovedXLocation,pieceToBeMovedYLocation,squareToBeMovedToXLocation,squareToBeMovedToYLocation


def reversePrediction(x,y,i,j):
    return (((x - 1) * 8 + y) - 1)  *64 + ((i - 1) * 8 + j)
    
    

def convertToAscii(board):
    
    for k in range(len(board)):
        for i in range(64):
            board[k][0][i] = ord(board[k][0][i])
            
            
def encode_board(board):
    """
    Encode the chess board into a 12-channel tensor.
    """
    # Define piece mappings
    piece_to_channel = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
    }
    
    # Initialize a 12-channel tensor
    encoded_board = np.zeros((8, 8, 12), dtype=np.float32)
    
    # Populate the tensor
    for i in range(8):
        for j in range(8):
            piece = board.piece_at(chess.square(j, 7-i))  # Note: chess.square expects (file, rank) with 0-indexed file
            if piece:
                channel = piece_to_channel[piece.symbol()]
                encoded_board[i, j, channel] = 1.0
    
    return encoded_board

def engineMove(self,x,y,i,j,promotionPiece):

    filteredPrediction = [0]*4096
    
    x = chr(x + 96)
    y = str(y)
    i = chr(i + 96)
    j = str(j)
    
    if (not(promotionPiece == None)):
        print(x,y,i,j,promotionPiece)
        pgnBoard.push(move.from_uci(x+y+i+j+promotionPiece))
        print("WAAA")
    else:
        print(x,y,i,j)
        pgnBoard.push(move.from_uci(x+y+i+j))
    
    t0= timer()
   
    inputBoard = [encode_board(pgnBoard)]
    #activeModel = keras.models.Sequential()
    prediction = model.predict(np.array(inputBoard))
    #printBoard(reverse(q))
    
    
    isLegal = False
    isCapture = False
    moves = []
    boardCopy = copy.deepcopy(self.boardPieces)
    colour = "Black"
    self.isComputerMove = True
    self.computerThinking = True
    
    # Create a pre-move copy of the board
    boardCopy = copy.deepcopy(self.boardPieces)
    
    # Loop through all pieces on the board
    for x in self.boardPieces:
        for y in x:
            
            # Check if the chosen piece is of the active player
            if (y.colour == colour):
                
                # Acquire list of possible moves for the given piece
                moves = []
                moveAppender(self, moves,y,colour)
    
                # Loop through list of moves
                for item in moves:
                    
                    # Set the piece to be moved as the current piece                        
                    self.pieceToBeMoved.piece = self.boardPieces[y.xLocation - 1][y.yLocation - 1].piece
                    self.pieceToBeMoved.colour = self.boardPieces[y.xLocation - 1][y.yLocation - 1].colour
                    self.pieceToBeMoved.value = self.boardPieces[y.xLocation - 1][y.yLocation - 1].value
                    self.pieceToBeMoved.xLocation = y.xLocation
                    self.pieceToBeMoved.yLocation = y.yLocation
                    print(self.boardPieces[4][6].piece, self.boardPieces[4][6].colour)
                    # Check if moving this piece to each move's location is legal
                    isLegal,isCapture = Rules.isLegalMove(self, self.boardPieces[item.xLocation - 1][item.yLocation - 1],item.xLocation,item.yLocation,colour)
                    self.isCastle = False
                    if(isLegal):
                        
                        self.boardPieces[item.xLocation - 1][item.yLocation - 1].piece = item.piece
                        self.boardPieces[item.xLocation - 1][item.yLocation - 1].colour = item.colour
                        self.boardPieces[item.xLocation - 1][item.yLocation - 1].value = item.value
                        
                        self.boardPieces[y.xLocation - 1][y.yLocation - 1].piece = "Empty"
                        self.boardPieces[y.xLocation - 1][y.yLocation - 1].colour = "None"
                        self.boardPieces[y.xLocation - 1][y.yLocation - 1].value = 0
                        
                        if(not(Rules.isInCheck(self, colour))):
                            filteredPrediction[reversePrediction(y.xLocation,y.yLocation,item.xLocation,item.yLocation) - 1] = prediction[0][reversePrediction(y.xLocation,y.yLocation,item.xLocation,item.yLocation) - 1]
                        self.boardPieces = copy.deepcopy(boardCopy)
               
    self.computerThinking = False                        
    self.boardPieces = copy.deepcopy(boardCopy)
    filteredPrediction = np.array(filteredPrediction)
    #filteredPrediction = np.argsort(filteredPrediction,axis = 2)
    print(np.argmax(filteredPrediction))
    #filteredPrediction[0][0][np.argmax(filteredPrediction)] = 0
    #print(np.argmax(filteredPrediction))                            
    
    #print(filteredPrediction[0][:,0])
    print(prediction[0][np.argmax(filteredPrediction)] * 100,"%")
    print(filteredPrediction[np.argmax(filteredPrediction)] * 100,"%")
    a,b,c,d = predictionInfo(np.argmax(filteredPrediction) + 1)
    
                        
    
    print("X1: ",a)
    print("Y1: ",b)
    print("X2: ",c)
    print("Y2: ",d)
    
    print()
    print (filteredPrediction.shape)
    for i in range (5):
        filteredPrediction[np.argmax(filteredPrediction)] = 0
        q,w,e,r = predictionInfo(np.argmax(filteredPrediction) + 1)   
    
        print("X1: ",q)
        print("Y1: ",w)
        print("X2: ",e)
        print("Y2: ",r)
        
        print(prediction[0][np.argmax(filteredPrediction)] * 100,"%")
        print()
        
    print(reversePrediction(a,b,c,d))
    pieceToBeMoved = self.boardPieces[a-1][b-1]
    currentItem = self.boardPieces[c-1][d-1]
    if(not(pieceToBeMoved == None) and reversePrediction(a,b,c,d) > 1):
        print(str(pieceToBeMoved.xLocation) + " " + str(pieceToBeMoved.yLocation) + " " + self.boardPieces[pieceToBeMoved.xLocation-1][pieceToBeMoved.yLocation-1].colour + self.boardPieces[pieceToBeMoved.xLocation-1][pieceToBeMoved.yLocation-1].piece)
        print(str(currentItem.xLocation) + " " + str(currentItem.yLocation) + " " + self.boardPieces[currentItem.xLocation-1][currentItem.yLocation-1].colour + self.boardPieces[currentItem.xLocation-1][currentItem.yLocation-1].piece)
        
        
        self.count = 0
        
        self.pieceToBeMoved.piece = pieceToBeMoved.piece
        self.pieceToBeMoved.colour = pieceToBeMoved.colour
        self.pieceToBeMoved.xLocation = pieceToBeMoved.xLocation
        self.pieceToBeMoved.yLocation = pieceToBeMoved.yLocation
        self.pieceToBeMoved.value = pieceToBeMoved.value
        self.move = True
        
        self.get_location(None, c, d)
        t1 = timer()
        print("Time elapsed: ", t1 - t0)
        print()
        a = chr(a + 96)
        b = str(b)
        c = chr(c + 96)
        d = str(d)

        if (pieceToBeMoved.piece == "Pawn" and b == 7 and d == 8):
            promotionPiece = 'q'
            print(a,b,c,d,promotionPiece)
            pgnBoard.push(move.from_uci(a+b+c+d+promotionPiece))
        else:
            print(a+b+c+d)
            print("ASDFFGHJ")
            pgnBoard.push(move.from_uci(a+b+c+d))
       
    else:
        easygui.msgbox("Black Resigns", title="Winner!")

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