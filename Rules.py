# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 20:53:20 2024

@author: Ranuja
"""

import copy
import Engine
#from numba import njit
   
# Function to define legal pawn moves
def isLegalPawnMove(self,curItem,i,j,colour):
    isLegal = False
    isCapture = False
    piece = None
    # Check if the pawn is moving to an empty location
    if(curItem.piece == "Empty"):
        if (colour == "White"):
            
            # Check if the white pawn is at its starting position
            if(self.pieceToBeMoved.yLocation == 2):
                
                # From here check if the pawn is moving either 1 square or 2, then check if there are any pieces impeding its path
                if(self.pieceToBeMoved.xLocation == i and (j == 3 or j == 4)):
                    if(j == 3 and self.boardPieces[(i-1, 2)].piece == "Empty"):
                        isLegal = True
                    elif(j == 4 and self.boardPieces[(i-1, 2)].piece == "Empty" and self.boardPieces[(i-1, 3)].piece == "Empty"):
                        isLegal = True
            
            # Move the pawn 1 up
            elif(self.pieceToBeMoved.xLocation == i and j - 1 == self.pieceToBeMoved.yLocation):
                isLegal = True
                
            # Check if the attempted move is an en pasent move
            elif(self.blackSideEnPasent and abs(self.pieceToBeMoved.xLocation - self.blackSideEnPasentPawnxLocation) == 1 and (i + 1 == self.pieceToBeMoved.xLocation and j - 1 == self.pieceToBeMoved.yLocation or i - 1 == self.pieceToBeMoved.xLocation and j - 1 == self.pieceToBeMoved.yLocation)):
                
                isLegal = True
                isCapture = True
            
            # Set the image variable    
            if (isLegal):    
                if (self.findSquareColour(i,j) =="White"):   
                    piece = self.canvas.whitePawnWhiteSquareImage
                else:
                    piece = self.canvas.whitePawnBlackSquareImage
        else:
            
            # Check if the black pawn is at its starting position
            if(self.pieceToBeMoved.yLocation == 7):
                
                # From here check if the pawn is moving either 1 square or 2, then check if there are any pieces impeding its path
                if(self.pieceToBeMoved.xLocation == i and (j == 6 or j == 5)):
                     if(j == 6 and self.boardPieces[(i-1, 5)].piece == "Empty"):
                        isLegal = True
                     elif(j == 5 and self.boardPieces[(i-1, 5)].piece == "Empty" and self.boardPieces[(i-1, 5)].piece == "Empty"):
                        isLegal = True
                        
            # Move the pawn 1 down
            elif(self.pieceToBeMoved.xLocation == i and j + 1 == self.pieceToBeMoved.yLocation):
                isLegal = True
                
            # Check if the attempted move is an en pasent move
            elif(self.whiteSideEnPasent and abs(self.pieceToBeMoved.xLocation - self.whiteSideEnPasentPawnxLocation) == 1 and (i + 1 == self.pieceToBeMoved.xLocation and j + 1 == self.pieceToBeMoved.yLocation or i - 1 == self.pieceToBeMoved.xLocation and j + 1 == self.pieceToBeMoved.yLocation)):
                isLegal = True
                isCapture = True 
            
            # Set the image variable  
            if (isLegal):    
                if (self.findSquareColour(i,j) =="White"):   
                    piece = self.canvas.blackPawnWhiteSquareImage
                else:
                    piece = self.canvas.blackPawnBlackSquareImage
    
    # Handle captures
    else:

        if (colour == "White"):
            
            # Check if the move is a legal capture
            if(curItem.colour == "Black" and (i + 1 == self.pieceToBeMoved.xLocation and j - 1 == self.pieceToBeMoved.yLocation or i - 1 == self.pieceToBeMoved.xLocation and j - 1 == self.pieceToBeMoved.yLocation)):
            
                # Set the image variable 
                if (self.findSquareColour(i,j) =="White"):   
                    piece = self.canvas.whitePawnWhiteSquareImage
                else:
                    piece = self.canvas.whitePawnBlackSquareImage
                isLegal = True
                isCapture = True
        else:
            
            # Check if the move is a legal capture    
            if(curItem.colour == "White" and (i + 1 == self.pieceToBeMoved.xLocation and j + 1 == self.pieceToBeMoved.yLocation or i - 1 == self.pieceToBeMoved.xLocation and j + 1 == self.pieceToBeMoved.yLocation)):
                
                # Check if the move is a legal capture
                if (self.findSquareColour(i,j) =="White"):   
                    piece = self.canvas.blackPawnWhiteSquareImage
                else:
                    piece = self.canvas.blackPawnBlackSquareImage
                isLegal = True
                isCapture = True

    return piece,isLegal,isCapture

# Function to define legal knight moves    
def isLegalKnightMove(self,curItem,i,j,colour):
    isLegal = False
    isCapture = False
    piece = None
    
    # Check if the destination is a knights move away
    if (abs(i - self.pieceToBeMoved.xLocation) == 2 and abs(j - self.pieceToBeMoved.yLocation) == 1 or abs(i - self.pieceToBeMoved.xLocation) == 1 and abs(j - self.pieceToBeMoved.yLocation) == 2):
        
        # Set the image variable and set isLegal as true
        if (colour == "White"):
            
            if (self.findSquareColour(i,j) =="White"):   
                piece = self.canvas.whiteKnightWhiteSquareImage
            else:
                piece = self.canvas.whiteKnightBlackSquareImage
            isLegal = True
            
            # Check if the knight is capturing a piece
            if(curItem.colour == "Black"):
                isCapture = True
        else:
            
            if (self.findSquareColour(i,j) =="White"):   
                piece = self.canvas.blackKnightWhiteSquareImage
            else:
                piece = self.canvas.blackKnightBlackSquareImage
            isLegal = True
            
            # Check if the knight is capturing a piece
            if(curItem.colour == "White"):
                isCapture = True

    return piece,isLegal,isCapture

# Function to define legal bishop moves
def isLegalBishopMove(self,curItem,i,j,colour):
    
    isLegal = False
    isCapture = False
    piece = None
    
    if (abs(i - self.pieceToBeMoved.xLocation) ==abs(j - self.pieceToBeMoved.yLocation)):
        
        xMultiplier = 1
        yMultiplier = 1
        isLegal = True
        
        # If the defending piece is ahead of the attacking, the check must be done in the other direction
        if(i > self.pieceToBeMoved.xLocation):
            xMultiplier = -1
        if (j > self.pieceToBeMoved.yLocation):
            yMultiplier = -1
        
        # Check if there is an impeding piece in the way, if so the attacker is not attacking the defending piece
        for square in range(1,abs(i - self.pieceToBeMoved.xLocation)):
            
            if(not(self.boardPieces[(i + square*xMultiplier - 1, j + square*yMultiplier - 1)].piece == "Empty")):
                isLegal = False
        
        # Set the image variable and set isLegal as true
        if(isLegal):
            isLegal = False
            if (colour == "White"):
                
                if (self.findSquareColour(i,j) =="White"):   
                    piece = self.canvas.whiteBishopWhiteSquareImage
                else:
                    piece = self.canvas.whiteBishopBlackSquareImage
                isLegal = True
                
                # Check if the bishop is capturing a piece
                if(curItem.colour == "Black"):
                    isCapture = True   
            else:
                
                if (self.findSquareColour(i,j) =="White"):   
                    piece = self.canvas.blackBishopWhiteSquareImage
                else:
                    piece = self.canvas.blackBishopBlackSquareImage
                isLegal = True
                
                # Check if the bishop is capturing a piece
                if(curItem.colour == "White"):
                    isCapture = True  

    return piece,isLegal,isCapture

# Function to define legal rook moves
def isLegalRookMove(self,curItem,i,j,colour):
    isLegal = False
    isCapture = False
    piece = None
    #print(self.pieceToBeMoved.xLocation,self.pieceToBeMoved.yLocation)
    if(i == self.pieceToBeMoved.xLocation or j == self.pieceToBeMoved.yLocation):
        
        xMultiplier = 1
        yMultiplier = 1
        isLegal = True
        
        # Vertical moves
        if (i == self.pieceToBeMoved.xLocation):
            
            # If the defending piece is ahead of the attacking, the check must be done in the other direction
            if (j > self.pieceToBeMoved.yLocation):
                yMultiplier = -1
                
            # Check if there is an impeding piece in the way, if so the attacker is not attacking the defending piece
            for square in range(1,abs(j - self.pieceToBeMoved.yLocation)):                
                if(not(self.boardPieces[(i-1, j + square*yMultiplier - 1)].piece == "Empty")):
                    isLegal = False
        # Horizontal moves
        else:
            
            # If the defending piece is ahead of the attacking, the check must be done in the other direction
            if(i > self.pieceToBeMoved.xLocation):
                xMultiplier = -1
        
            # Check if there is an impeding piece in the way, if so the attacker is not attacking the defending piece
            for square in range(1,abs(i - self.pieceToBeMoved.xLocation)):
                if(not(self.boardPieces[(i+ square*xMultiplier - 1, j-1)].piece == "Empty")):
                    isLegal = False
        
        # Set the image variable and set isLegal as true
        if (isLegal):
            isLegal = False
            if (colour == "White"):
                
                if (self.findSquareColour(i,j) =="White"):   
                    piece = self.canvas.whiteRookWhiteSquareImage
                else:
                    piece = self.canvas.whiteRookBlackSquareImage
                isLegal = True
                
                # Check if the rook is capturing a piece
                if(curItem.colour == "Black"):
                    isCapture = True   
            else:
                
                if (self.findSquareColour(i,j) =="White"):   
                    piece = self.canvas.blackRookWhiteSquareImage
                else:
                    piece = self.canvas.blackRookBlackSquareImage
                isLegal = True
                
                # Check if the rook is capturing a piece
                if(curItem.colour == "White"):
                    isCapture = True 
                   
    return piece,isLegal,isCapture

# Function to define legal queen moves
def isLegalQueenMove(self,curItem,i,j,colour):
    isLegal = False
    isCapture = False
    piece = None
    # Set the default direction of exploration
    xMultiplier = 1
    yMultiplier = 1
    isLegal = True

    # Check along diagonals    
    if (abs(i - self.pieceToBeMoved.xLocation) ==abs(j - self.pieceToBeMoved.yLocation)):
        
        # If the defending piece is ahead of the attacking, the check must be done in the other direction
        if(i > self.pieceToBeMoved.xLocation):
            xMultiplier = -1
        if (j > self.pieceToBeMoved.yLocation):
            yMultiplier = -1
        
        # Check if there is an impeding piece in the way, if so the attacker is not attacking the defending piece
        for square in range(1,abs(i - self.pieceToBeMoved.xLocation)):
            
            if(not(self.boardPieces[(i + square*xMultiplier - 1, j + square*yMultiplier - 1)].piece == "Empty")):
                isLegal = False
                
    # Check along rows and columns
    elif(i == self.pieceToBeMoved.xLocation or j == self.pieceToBeMoved.yLocation):
        
        # Vertical moves
        if (i == self.pieceToBeMoved.xLocation):
            
            # If the defending piece is ahead of the attacking, the check must be done in the other direction
            if (j > self.pieceToBeMoved.yLocation):
                yMultiplier = -1
                
            # Check if there is an impeding piece in the way, if so the attacker is not attacking the defending piece
            for square in range(1,abs(j - self.pieceToBeMoved.yLocation)):                
                if(not(self.boardPieces[(i-1, j + square*yMultiplier - 1)].piece == "Empty")):
                    isLegal = False
        # Horizontal moves
        else:
            
            # If the defending piece is ahead of the attacking, the check must be done in the other direction
            if(i > self.pieceToBeMoved.xLocation):
                xMultiplier = -1
        
            # Check if there is an impeding piece in the way, if so the attacker is not attacking the defending piece
            for square in range(1,abs(i - self.pieceToBeMoved.xLocation)):
                if(not(self.boardPieces[(i + square*xMultiplier - 1, j-1)].piece == "Empty")):
                    isLegal = False
    else:
        isLegal = False
    
    # Set the image variable and set isLegal as true
    if (isLegal):
            isLegal = False
            if (colour == "White"):
                
                if (self.findSquareColour(i,j) =="White"):   
                    piece = self.canvas.whiteQueenWhiteSquareImage
                else:
                    piece = self.canvas.whiteQueenBlackSquareImage
                isLegal = True
                
                # Check if the queen is capturing a piece
                if(curItem.colour == "Black"):
                    isCapture = True   
            else:
                
                if (self.findSquareColour(i,j) =="White"):   
                    piece = self.canvas.blackQueenWhiteSquareImage
                else:
                    piece = self.canvas.blackQueenBlackSquareImage
                isLegal = True
                
                # Check if the queen is capturing a piece
                if(curItem.colour == "White"):
                    isCapture = True 
    
    return piece,isLegal,isCapture

# Function to define legal king moves
def isLegalKingMove(self,curItem,i,j,colour):
    isLegal = False
    isCapture = False
    piece = None
    
    if(abs(i - self.pieceToBeMoved.xLocation) < 2 and abs(j - self.pieceToBeMoved.yLocation) < 2):
        
        # Set the image variable and set isLegal as true
        if (colour == "White"):
            
            if (self.findSquareColour(i,j) =="White"):   
                piece = self.canvas.whiteKingWhiteSquareImage
            else:
                piece = self.canvas.whiteKingBlackSquareImage
            isLegal = True
            
            # Check if the king is capturing a piece
            if(curItem.colour == "Black"):
                isCapture = True   
        else:
            
            if (self.findSquareColour(i,j) =="White"):   
                piece = self.canvas.blackKingWhiteSquareImage
            else:
                piece = self.canvas.blackKingBlackSquareImage
            isLegal = True
            
            # Check if the king is capturing a piece
            if(curItem.colour == "White"):
                isCapture = True
    
    # Check for attempted castle
    else:
        isLegal = False
        if (colour == "White"):
            
            # White kingside castle
            if (i == 7 and j == 1):
                
                # Check that neither of the involved pieces have previously moved
                if(not(self.whiteKingHasMoved) and not(self.kingSideWhiteRookHasMoved)):
                    
                    # Check if there are no pieces obstructing castling
                    if (self.boardPieces[(5, 0)].piece == "Empty" and self.boardPieces[(6, 0)].piece == "Empty" and self.boardPieces[(7, 0)].piece == "Rook" and self.boardPieces[(7, 0)].colour == "White"):
                        self.isCastle = True
                        
                        # Check if any opposing piece is attacking the castling path
                        for y in self.boardPieces.values():
                            if (y.colour == "Black"):
                                if(isAttacking(self, y,self.boardPieces[(4, 0)]) or isAttacking(self, y,self.boardPieces[(5, 0)]) or isAttacking(self, y,self.boardPieces[(6, 0)])):
                                     self.isCastle = False
                                         
            # White queenside castle                             
            if (i == 3 and j == 1):
                
                # Check that neither of the involved pieces have previously moved
                if(not(self.whiteKingHasMoved) and not(self.queenSideWhiteRookHasMoved)):
                    
                    # Check if there are no pieces obstructing castling
                    if (self.boardPieces[(1, 0)].piece == "Empty" and self.boardPieces[(2, 0)].piece == "Empty" and self.boardPieces[(3, 0)].piece == "Empty" and self.boardPieces[(0, 0)].piece == "Rook" and self.boardPieces[(0, 0)].colour == "White"):
                        self.isCastle = True
                        
                        # Check if any opposing piece is attacking the castling path
                        for y in self.boardPieces.values():
                            if (y.colour == "Black"):
                                if(isAttacking(self, y, self.boardPieces[(4, 0)]) or isAttacking(self ,y,self.boardPieces[(1, 0)]) or isAttacking(self, y,self.boardPieces[(2, 0)]) or isAttacking(self, y,self.boardPieces[(3, 0)])):
                                     self.isCastle = False
                                         
        else:
            
            # Black kingside castle
            if (i == 7 and j == 8):
                
                # Check that neither of the involved pieces have previously moved
                if(not(self.blackKingHasMoved) and not(self.kingSideBlackRookHasMoved)):
                    
                    # Check if there are no pieces obstructing castling
                    if (self.boardPieces[(5, 7)].piece == "Empty" and self.boardPieces[(6, 7)].piece == "Empty" and self.boardPieces[(7, 7)].piece == "Rook" and self.boardPieces[(7, 7)].colour == "Black"):
                        self.isCastle = True
                        
                        # Check if any opposing piece is attacking the castling path
                        for y in self.boardPieces.values():
                            if (y.colour == "White"):
                                if(isAttacking(self, y,self.boardPieces[(4, 7)]) or isAttacking(self, y,self.boardPieces[(5, 7)]) or isAttacking(self, y,self.boardPieces[(6, 7)])):
                                     self.isCastle = False
            # Black queenside castle                             
            if (i == 3 and j == 8):
                
                # Check that neither of the involved pieces have previously moved
                if(not(self.blackKingHasMoved) and not(self.queenSideBlackRookHasMoved)):
                    
                    # Check if there are no pieces obstructing castling
                    if (self.boardPieces[(1, 7)].piece == "Empty" and self.boardPieces[(2, 7)].piece == "Empty" and self.boardPieces[(3, 7)].piece == "Empty" and self.boardPieces[(4, 7)].piece == "Rook" and self.boardPieces[(0, 7)].colour == "Black"):
                        self.isCastle = True
                        
                        # Check if any opposing piece is attacking the castling path
                        for y in self.boardPieces.values():
                            if (y.colour == "White"):
                                if(isAttacking(self, y,self.boardPieces[(4, 7)]) or isAttacking(self, y,self.boardPieces[(1, 7)]) or isAttacking(self, y,self.boardPieces[(2, 7)]) or isAttacking(self, y,self.boardPieces[(3, 7)])):
                                    self.isCastle = False
        
        if(self.isCastle):
            # Only if the move is actually to be made should the variables be set
            # Otherwise just set as legal for the engine's calculation
            isLegal = True
            if(not(self.computerThinking)):
            
                if(i == 7 and j == 1):
                    piece = self.canvas.whiteKingBlackSquareImage
                    self.whiteKingHasMoved = True
                    self.kingSideWhiteRookHasMoved = True
                    self.whiteHasCastled = True
                elif (i == 3 and j == 1):
                    piece = self.canvas.whiteKingBlackSquareImage
                    self.whiteKingHasMoved = True
                    self.queenSideWhiteRookHasMoved = True
                    self.whiteHasCastled = True
                elif (i == 7 and j == 8):
                    piece = self.canvas.blackKingWhiteSquareImage
                    self.blackKingHasMoved = True
                    self.kingSideBlackRookHasMoved = True
                    self.blackHasCastled = True
                elif (i == 3 and j == 8):
                    piece = self.canvas.blackKingWhiteSquareImage
                    self.blackKingHasMoved = True
                    self.queenSideBlackRookHasMoved = True
                    self.blackHasCastled = True
        
    return piece,isLegal,isCapture

# Function to check if a move is legal
def isLegalMove(self,curItem,i,j,colour):
    
    boardCopy = copy.deepcopy(self.boardPieces)
                         
    # Check the required piece's rules to see if the move is legal
    if(self.pieceToBeMoved.colour == colour and not(curItem.colour == colour) and not(curItem.piece == "King")):
        if(self.pieceToBeMoved.piece == "Pawn"):                
            piece,isLegal,isCapture = isLegalPawnMove(self, curItem,i,j,colour)     
        elif(self.pieceToBeMoved.piece == "Knight"):
            piece,isLegal,isCapture = isLegalKnightMove(self, curItem,i,j,colour) 
        elif(self.pieceToBeMoved.piece == "Bishop"):
            piece,isLegal,isCapture = isLegalBishopMove(self, curItem,i,j,colour) 
        elif(self.pieceToBeMoved.piece == "Rook"):
            piece,isLegal,isCapture = isLegalRookMove(self, curItem,i,j,colour)  
        elif(self.pieceToBeMoved.piece == "Queen"):
            piece,isLegal,isCapture = isLegalQueenMove(self, curItem,i,j,colour)                     
        elif(self.pieceToBeMoved.piece == "King"):
            piece,isLegal,isCapture = isLegalKingMove(self, curItem,i,j,colour)
        
        if (isLegal):
            
            # Make the location square hold the moving piece
            self.boardPieces[(i-1, j-1)].piece = self.pieceToBeMoved.piece
            self.boardPieces[(i-1, j-1)].colour = self.pieceToBeMoved.colour
            
            # Empty the previous square
            self.boardPieces[(self.pieceToBeMoved.xLocation - 1, self.pieceToBeMoved.yLocation-1)].piece = "Empty"
            self.boardPieces[(self.pieceToBeMoved.xLocation - 1, self.pieceToBeMoved.yLocation-1)].colour = "None"

            # Check if making this move puts the current player under check
            # If so, the move is not legal
            if (self.numMove % 2 == 0):
                if(isInCheck(self, "White")):
                    isLegal = False
            else:
                if(isInCheck(self, "Black")):    
                     isLegal = False
            
            self.boardPieces = copy.deepcopy(boardCopy)     
            
            if(self.move):
                return piece,isLegal,isCapture
            else:
                return isLegal,isCapture
        else:

            if(self.move):
                return piece,isLegal,isCapture
            else:
                return isLegal,isCapture        
    else:

        if(self.move):
            return None,False,False
        else:
            return False,False

# Function to find out if a player is in checkmate
def isCheckMate(self,colour):
    isLegal = False
    isCapture = False
    moves = []
    boardCopy = copy.deepcopy(self.boardPieces)
    
    # Makes all possible moves to see if it can avoid being in check
    for y in self.boardPieces.values():
        if (y.colour == colour and not(y.piece == "Empty")):
            
            # Acquire list of possible moves for the given piece
            moves = []
            Engine.moveAppender(self, moves,y,colour)
            
            # Loop through list of moves
            for item in moves:
                
                # Set the piece to be moved as the current piece                        
                self.pieceToBeMoved.piece = self.boardPieces[(y.xLocation - 1, y.yLocation - 1)].piece
                self.pieceToBeMoved.colour = self.boardPieces[(y.xLocation - 1, y.yLocation - 1)].colour
                self.pieceToBeMoved.value = self.boardPieces[(y.xLocation - 1, y.yLocation - 1)].value
                self.pieceToBeMoved.xLocation = y.xLocation
                self.pieceToBeMoved.yLocation = y.yLocation
                
                # Check if moving this piece to each move's location is legal
                isLegal,isCapture = isLegalMove(self, self.boardPieces[(item.xLocation - 1, item.yLocation - 1)],item.xLocation,item.yLocation,colour)

                if(isLegal):
                    
                    if(not(item.piece == y.piece)):
                        promotion = True
                    
                    self.boardPieces[(item.xLocation - 1, item.yLocation - 1)].piece = item.piece
                    self.boardPieces[(item.xLocation - 1, item.yLocation - 1)].colour = item.colour
                    self.boardPieces[(item.xLocation - 1, item.yLocation - 1)].value = item.value
                    
                    self.boardPieces[(y.xLocation - 1, y.yLocation - 1)].piece = "Empty"
                    self.boardPieces[(y.xLocation - 1, y.yLocation - 1)].colour = "None"
                    self.boardPieces[(y.xLocation - 1, y.yLocation - 1)].value = 0
                    
                    # Check if the player is still in check after the move
                    if(not(isInCheck(self, colour))):
                        
                        # If the player is not in check after a move, then the player is not in checkmate
                        self.boardPieces = copy.deepcopy(boardCopy)
                        return False
                    self.boardPieces = copy.deepcopy(boardCopy)
    return True        
    
# Function to check if the given side is in check        
def isInCheck(self, colour):
    # Find the king piece of the given colour
    kingPiece = next(
        (piece for piece in self.boardPieces.values() if piece.piece == "King" and piece.colour == colour), 
        None
    )    
                       
    # Check if any piece is attacking the king
    for piece in self.boardPieces.values():
        if piece.colour != colour and piece.piece != "Empty":
            if isAttacking(self, piece, kingPiece):
                return True
                
    return False
    
    
# Function to check if a piece if attacking another piece
def isAttacking (self,attacker,defender):
    curColour = attacker.colour
    
    self.count+= 1
    isLegal = False
    
    if (attacker.piece == "Pawn"):
        try:
            # Check either side of the defender
            if (curColour == "White"):
                if((defender.xLocation + 1 == attacker.xLocation and defender.yLocation - 1 == attacker.yLocation or defender.xLocation - 1 == attacker.xLocation and defender.yLocation - 1 == attacker.yLocation)):
                    isLegal = True
      
            # Check either side of the defender
            elif(curColour == "Black"):
                if((defender.xLocation + 1 == attacker.xLocation and defender.yLocation + 1 == attacker.yLocation or defender.xLocation - 1 == attacker.xLocation and defender.yLocation + 1 == attacker.yLocation)):
                    isLegal = True
        except:
            pass
        
    # Check knight moves
    elif(attacker.piece == "Knight"):
        try:
            if (abs(defender.xLocation - attacker.xLocation) == 2 and abs(defender.yLocation - attacker.yLocation) == 1 or abs(defender.xLocation - attacker.xLocation) == 1 and abs(defender.yLocation - attacker.yLocation) == 2):
                isLegal = True
        except:
            pass
          
    elif(attacker.piece == "Bishop"):
        
        # Check along diagonals
        if (abs(defender.xLocation - attacker.xLocation) == abs(defender.yLocation - attacker.yLocation)):
            
            xMultiplier = 1
            yMultiplier = 1
            
            # If the defending piece is ahead of the attacking, the check must be done in the other direction
            if(defender.xLocation > attacker.xLocation):
                xMultiplier = -1
            if (defender.yLocation > attacker.yLocation):
                yMultiplier = -1
            
            # Check if there is an impeding piece in the way, if so the attacker is not attacking the defending piece
            for square in range(1,abs(defender.xLocation - attacker.xLocation)):
                if(not(self.boardPieces[(defender.xLocation + square*xMultiplier - 1, defender.yLocation + square*yMultiplier - 1)].piece == "Empty")):
                    return False
                
            isLegal = True
        
    elif(attacker.piece == "Rook"):
        
        if(defender.xLocation == attacker.xLocation or defender.yLocation == attacker.yLocation):
                
            xMultiplier = 1
            yMultiplier = 1
            
            # Vertical moves
            if (defender.xLocation == attacker.xLocation):
                
                # If the defending piece is ahead of the attacking, the check must be done in the other direction
                if (defender.yLocation > attacker.yLocation):
                    yMultiplier = -1
                    
                # Check if there is an impeding piece in the way, if so the attacker is not attacking the defending piece    
                for square in range(1,abs(defender.yLocation - attacker.yLocation)):
                    if(not(self.boardPieces[(defender.xLocation - 1, defender.yLocation + square*yMultiplier - 1)].piece == "Empty")):
                        return False
            
            # Horizontal moves   
            else:
                
                # If the defending piece is ahead of the attacking, the check must be done in the other direction
                if(defender.xLocation > attacker.xLocation):
                    xMultiplier = -1
            
                # Check if there is an impeding piece in the way, if so the attacker is not attacking the defending piece 
                for square in range(1,abs(defender.xLocation - attacker.xLocation)):
                    if(not(self.boardPieces[(defender.xLocation + square*xMultiplier - 1, defender.yLocation - 1)].piece == "Empty")):
                        return False
                    
            isLegal = True           
        
    elif(attacker.piece == "Queen"):
        
        # Set the default direction of exploration
        xMultiplier = 1
        yMultiplier = 1
        
        # Check along diagonals
        if (abs(defender.xLocation - attacker.xLocation) == abs(defender.yLocation - attacker.yLocation)):
            
            # If the defending piece is ahead of the attacking, the check must be done in the other direction
            if(defender.xLocation > attacker.xLocation):
                xMultiplier = -1
            if (defender.yLocation > attacker.yLocation):
                yMultiplier = -1

            # Check if there is an impeding piece in the way, if so the attacker is not attacking the defending piece
            for square in range(1,abs(defender.xLocation - attacker.xLocation)):
                if(not(self.boardPieces[(defender.xLocation + square*xMultiplier - 1, defender.yLocation + square*yMultiplier - 1)].piece == "Empty")):
                    return False
        
        # Check along rows and columns
        elif(defender.xLocation == attacker.xLocation or defender.yLocation == attacker.yLocation):
            
            # Vertical moves
            if (defender.xLocation == attacker.xLocation):
                
                # If the defending piece is ahead of the attacking, the check must be done in the other direction
                if (defender.yLocation > attacker.yLocation):
                    yMultiplier = -1
                    
                # Check if there is an impeding piece in the way, if so the attacker is not attacking the defending piece    
                for square in range(1,abs(defender.yLocation - attacker.yLocation)):
                    if(not(self.boardPieces[(defender.xLocation - 1, defender.yLocation + square*yMultiplier - 1)].piece == "Empty")):
                        return False
            
            # Horizontal moves   
            else:
                
                # If the defending piece is ahead of the attacking, the check must be done in the other direction
                if(defender.xLocation > attacker.xLocation):
                    xMultiplier = -1
            
                # Check if there is an impeding piece in the way, if so the attacker is not attacking the defending piece 
                for square in range(1,abs(defender.xLocation - attacker.xLocation)):
                    if(not(self.boardPieces[(defender.xLocation + square*xMultiplier - 1, defender.yLocation - 1)].piece == "Empty")):
                        return False
        else:
            return False
        
        isLegal = True
        
    elif(attacker.piece == "King"):
        # Check if the vertical and horizontal distance between the two pieces is 1 or less
        if(abs(defender.xLocation - attacker.xLocation) < 2 and abs(defender.yLocation - attacker.yLocation) < 2):
            isLegal = True
        
    if (isLegal):
        
        # Must check if the act of attacking would leave the attacker in check
        # This would mean that the defender is not actually under attack
        if (not(defender.piece == "King")):
            boardCopy = copy.deepcopy(self.boardPieces)

            # Make the location square hold the moving piece
            self.boardPieces[(defender.xLocation - 1, defender.yLocation - 1)].piece = attacker.piece
            self.boardPieces[(defender.xLocation - 1, defender.yLocation - 1)].colour = attacker.colour
            
            self.boardPieces[(attacker.xLocation - 1, attacker.yLocation - 1)].piece = "Empty"
            self.boardPieces[(attacker.xLocation - 1, attacker.yLocation - 1)].colour = "None"
             
            if (attacker.colour == "White"):
                 
                if(isInCheck(self, "White")):
                    self.boardPieces = copy.deepcopy(boardCopy)
                    return False
            else:
                if(isInCheck(self, "Black")):  
                     self.boardPieces = copy.deepcopy(boardCopy)
                     return False
             
                
            self.boardPieces = copy.deepcopy(boardCopy)  
        return True
    return False