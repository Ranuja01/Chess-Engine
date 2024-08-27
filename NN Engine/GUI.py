import tkinter as tk
import copy 
import easygui
import random
from timeit import default_timer as timer
import chess
import platform
#from numba import njit

from PIL import ImageTk, Image

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf

import Rules
#import Engine
import NNEngine

# Class to define a piece
class Pieces():
    
    def __init__(self,pieceType,colour,x,y):
        self.piece = pieceType
        self.colour = colour
        self.xLocation = x
        self.yLocation = y
        self.value = 0
        
        if(self.piece == "Pawn"):
            self.value = 1000 
        elif(self.piece == "Knight"):
            self.value = 2700  
        elif(self.piece == "Bishop"):
            self.value = 3000 
        elif(self.piece == "Rook"):
            self.value = 5000   
        elif(self.piece == "Queen"):
            self.value = 9000 
        elif(self.piece == "King"):
            self.value = 0                      

class Layout(tk.Tk):
    colours = ["#ffffff", "#a72f03"]#square colours light then dark
            
    def __init__(self, n=8):
        super().__init__()
        
        # Variables for drawing the board graphically
        self.n = n
        self.leftframe = tk.Frame(self)
        self.leftframe.grid(row=0, column=0, rowspan=10, padx=40)
        self.middleframe = tk.Frame(self)
        self.middleframe.grid(row=0, column=8, rowspan=8)
        self.canvas = tk.Canvas(self, width=950, height=768, )
        self.canvas.grid(row=0, column=1, columnspan=8, rowspan=8)
        self.board = [[None for row in range(n)] for col in range(n)]
        
        # Variables to flag movement and count the current move
        self.move = False
        self.numMove = 0
        
        # Variables to hold the moves selected
        self.pieceToBeMoved = Pieces("Empty","Empty",0,0)
        self.pawnToBePromoted = Pieces("Empty","Empty",0,0)
        
        # Variables to help with engine calculation
        self.isComputerMove = False
        self.computerThinking = False
        self.depth = 4
        self.pieceChosen = "None"
        
        # Variables to hold castling status
        self.whiteKingHasMoved = False
        self.blackKingHasMoved = False
        self.queenSideWhiteRookHasMoved = False
        self.kingSideWhiteRookHasMoved = False
        self.queenSideBlackRookHasMoved = False
        self.kingSideBlackRookHasMoved = False
        self.blackHasCastled = False
        self.whiteHasCastled = False
        self.isCastle = False
        
        # Variables to hold en pasent info
        self.blackSideEnPasent = False
        self.blackSideEnPasentPawnxLocation = 0
        self.whiteSideEnPasent = False
        self.whiteSideEnPasentPawnxLocation = 0

        # Promotion flag
        self.isPromotion = False
        
        # Check flag
        self.isCheck = False
        
        # Calculation count
        self.count = 0
        
        self.boardPieces = [[Pieces("Rook","White",1,1),
                            Pieces("Pawn","White",1,2),
                            Pieces("Empty","None",1,3),
                            Pieces("Empty","None",1,4),
                            Pieces("Empty","None",1,5),
                            Pieces("Empty","None",1,6),
                            Pieces("Pawn","Black",1,7),
                            Pieces("Rook","Black",1,8)],
                            
                            [Pieces("Knight","White",2,1),
                            Pieces("Pawn","White",2,2),
                            Pieces("Empty","None",2,3),
                            Pieces("Empty","None",2,4),
                            Pieces("Empty","None",2,5),
                            Pieces("Empty","None",2,6),
                            Pieces("Pawn","Black",2,7),
                            Pieces("Knight","Black",2,8)],
                            
                            [Pieces("Bishop","White",3,1),
                            Pieces("Pawn","White",3,2),
                            Pieces("Empty","None",3,3),
                            Pieces("Empty","None",3,4),
                            Pieces("Empty","None",3,5),
                            Pieces("Empty","None",3,6),
                            Pieces("Pawn","Black",3,7),
                            Pieces("Bishop","Black",3,8)],
                            
                            [Pieces("Queen","White",4,1),
                            Pieces("Pawn","White",4,2),
                            Pieces("Empty","None",4,3),
                            Pieces("Empty","None",4,4),
                            Pieces("Empty","None",4,5),
                            Pieces("Empty","None",4,6),
                            Pieces("Pawn","Black",4,7),
                            Pieces("Queen","Black",4,8)],
                            
                            [Pieces("King","White",5,1),
                            Pieces("Pawn","White",5,2),
                            Pieces("Empty","None",5,3),
                            Pieces("Empty","None",5,4),
                            Pieces("Empty","None",5,5),
                            Pieces("Empty","None",5,6),
                            Pieces("Pawn","Black",5,7),
                            Pieces("King","Black",5,8)],
                            
                            [Pieces("Bishop","White",6,1),
                            Pieces("Pawn","White",6,2),
                            Pieces("Empty","None",6,3),
                            Pieces("Empty","None",6,4),
                            Pieces("Empty","None",6,5),
                            Pieces("Empty","None",6,6),
                            Pieces("Pawn","Black",6,7),
                            Pieces("Bishop","Black",6,8)],
                            
                            [Pieces("Knight","White",7,1),
                            Pieces("Pawn","White",7,2),
                            Pieces("Empty","None",7,3),
                            Pieces("Empty","None",7,4),
                            Pieces("Empty","None",7,5),
                            Pieces("Empty","None",7,6),
                            Pieces("Pawn","Black",7,7),
                            Pieces("Knight","Black",7,8)],
                            
                            [Pieces("Rook","White",8,1),
                            Pieces("Pawn","White",8,2),
                            Pieces("Empty","None",8,3),
                            Pieces("Empty","None",8,4),
                            Pieces("Empty","None",8,5),
                            Pieces("Empty","None",8,6),
                            Pieces("Pawn","Black",8,7),
                            Pieces("Rook","Black",8,8)]
                          ]
        
    # Function to draw the board and its pieces graphically    
    def drawboard(self):
        
        # Draw the board itself
        from itertools import cycle
        for col in range(self.n):
            color = cycle(self.colours[::-1] if not col % 2 else self.colours)
            #print(color)
            for row in range(self.n):
                x1 = col * 90
                y1 = (7-row) * 90
                x2 = x1 + 90
                y2 = y1 + 90
                self.board[row][col] = self.canvas.create_rectangle(x1, y1, x2, y2, fill=next(color), tags=f"tile{col+1}{row+1}")
                self.canvas.tag_bind(f"tile{col+1}{row+1}","<Button-1>", lambda e, i=col+1, j=row+1: self.get_location(e,i,j))

        # Read all the required images from the file
        
        if platform.system() == 'Windows':
            blackPawnWhiteSquare = Image.open('../Images/BlackPawnWhiteSquare.png')
            blackPawnWhiteSquare = blackPawnWhiteSquare.resize((90, 90), Image.Resampling.LANCZOS)
            self.canvas.blackPawnWhiteSquareImage = ImageTk.PhotoImage(blackPawnWhiteSquare,master = self)
            
            blackPawnBlackSquare = Image.open('../Images/BlackPawnBlackSquare.png')
            blackPawnBlackSquare = blackPawnBlackSquare.resize((90, 90), Image.Resampling.LANCZOS)
            self.canvas.blackPawnBlackSquareImage = ImageTk.PhotoImage(blackPawnBlackSquare,master = self)
            
            blackRookWhiteSquare = Image.open('../Images/BlackRookWhiteSquare.png')
            blackRookWhiteSquare = blackRookWhiteSquare.resize((90, 90), Image.Resampling.LANCZOS)
            self.canvas.blackRookWhiteSquareImage = ImageTk.PhotoImage(blackRookWhiteSquare,master = self)
            
            blackKingWhiteSquare = Image.open('../Images/BlackKingWhiteSquare.png')
            blackKingWhiteSquare = blackKingWhiteSquare.resize((90, 90), Image.Resampling.LANCZOS)
            self.canvas.blackKingWhiteSquareImage = ImageTk.PhotoImage(blackKingWhiteSquare,master = self)
            
            blackKingBlackSquare = Image.open('../Images/BlackKingBlackSquare.png')
            blackKingBlackSquare = blackKingBlackSquare.resize((90, 90), Image.Resampling.LANCZOS)
            self.canvas.blackKingBlackSquareImage = ImageTk.PhotoImage(blackKingBlackSquare,master = self)
            
            blackKnightWhiteSquare = Image.open('../Images/BlackKnightWhiteSquare.png')
            blackKnightWhiteSquare = blackKnightWhiteSquare.resize((90, 90), Image.Resampling.LANCZOS)
            self.canvas.blackKnightWhiteSquareImage = ImageTk.PhotoImage(blackKnightWhiteSquare,master = self)
            
            blackKnightBlackSquare = Image.open('../Images/BlackKnightBlackSquare.png')
            blackKnightBlackSquare = blackKnightBlackSquare.resize((90, 90), Image.Resampling.LANCZOS)
            self.canvas.blackKnightBlackSquareImage = ImageTk.PhotoImage(blackKnightBlackSquare,master = self)
            
            blackRookBlackSquare = Image.open('../Images/BlackRookBlackSquare.png')
            blackRookBlackSquare = blackRookBlackSquare.resize((90, 90), Image.Resampling.LANCZOS)
            self.canvas.blackRookBlackSquareImage = ImageTk.PhotoImage(blackRookBlackSquare,master = self)
            
            blackQueenBlackSquare = Image.open('../Images/BlackQueenBlackSquare.png')
            blackQueenBlackSquare = blackQueenBlackSquare.resize((90, 90), Image.Resampling.LANCZOS)
            self.canvas.blackQueenBlackSquareImage = ImageTk.PhotoImage(blackQueenBlackSquare,master = self)
            
            blackQueenWhiteSquare = Image.open('../Images/BlackQueenWhiteSquare.png')
            blackQueenWhiteSquare = blackQueenWhiteSquare.resize((90, 90), Image.Resampling.LANCZOS)
            self.canvas.blackQueenWhiteSquareImage = ImageTk.PhotoImage(blackQueenWhiteSquare,master = self)
            
            blackBishopWhiteSquare = Image.open('../Images/BlackBishopWhiteSquare.png')
            blackBishopWhiteSquare = blackBishopWhiteSquare.resize((90, 90), Image.Resampling.LANCZOS)
            self.canvas.blackBishopWhiteSquareImage = ImageTk.PhotoImage(blackBishopWhiteSquare,master = self)
              
            blackBishopBlackSquare = Image.open('../Images/BlackBishopBlackSquare.png')
            blackBishopBlackSquare = blackBishopBlackSquare.resize((90, 90), Image.Resampling.LANCZOS)
            self.canvas.blackBishopBlackSquareImage = ImageTk.PhotoImage(blackBishopBlackSquare,master = self) 
            
            whitePawnWhiteSquare = Image.open('../Images/whitePawnWhiteSquare.png')
            whitePawnWhiteSquare = whitePawnWhiteSquare.resize((90, 90), Image.Resampling.LANCZOS)
            self.canvas.whitePawnWhiteSquareImage = ImageTk.PhotoImage(whitePawnWhiteSquare,master = self)

            whitePawnBlackSquare = Image.open('../Images/whitePawnBlackSquare.png')
            whitePawnBlackSquare = whitePawnBlackSquare.resize((90, 90), Image.Resampling.LANCZOS)
            self.canvas.whitePawnBlackSquareImage = ImageTk.PhotoImage(whitePawnBlackSquare,master = self)
            
            whiteRookWhiteSquare = Image.open('../Images/whiteRookWhiteSquare.png')
            whiteRookWhiteSquare = whiteRookWhiteSquare.resize((90, 90), Image.Resampling.LANCZOS)
            self.canvas.whiteRookWhiteSquareImage = ImageTk.PhotoImage(whiteRookWhiteSquare,master = self)
            
            whiteRookBlackSquare = Image.open('../Images/whiteRookBlackSquare.png')
            whiteRookBlackSquare = whiteRookBlackSquare.resize((90, 90), Image.Resampling.LANCZOS)
            self.canvas.whiteRookBlackSquareImage = ImageTk.PhotoImage(whiteRookBlackSquare,master = self)
            
            whiteKnightWhiteSquare = Image.open('../Images/whiteKnightWhiteSquare.png')
            whiteKnightWhiteSquare = whiteKnightWhiteSquare.resize((90, 90), Image.Resampling.LANCZOS)
            self.canvas.whiteKnightWhiteSquareImage = ImageTk.PhotoImage(whiteKnightWhiteSquare,master = self)
            
            whiteKnightBlackSquare = Image.open('../Images/whiteKnightBlackSquare.png')
            whiteKnightBlackSquare = whiteKnightBlackSquare.resize((90, 90), Image.Resampling.LANCZOS)
            self.canvas.whiteKnightBlackSquareImage = ImageTk.PhotoImage(whiteKnightBlackSquare,master = self)
            
            whiteKingBlackSquare = Image.open('../Images/whiteKingBlackSquare.png')
            whiteKingBlackSquare = whiteKingBlackSquare.resize((90, 90), Image.Resampling.LANCZOS)
            self.canvas.whiteKingBlackSquareImage = ImageTk.PhotoImage(whiteKingBlackSquare,master = self)
            
            whiteBishopWhiteSquare = Image.open('../Images/whiteBishopWhiteSquare.png')
            whiteBishopWhiteSquare = whiteBishopWhiteSquare.resize((90, 90), Image.Resampling.LANCZOS)
            self.canvas.whiteBishopWhiteSquareImage = ImageTk.PhotoImage(whiteBishopWhiteSquare,master = self)
            
            whiteBishopBlackSquare = Image.open('../Images/whiteBishopBlackSquare.png')
            whiteBishopBlackSquare = whiteBishopBlackSquare.resize((90, 90), Image.Resampling.LANCZOS)
            self.canvas.whiteBishopBlackSquareImage = ImageTk.PhotoImage(whiteBishopBlackSquare,master = self)
            
            whiteQueenWhiteSquare = Image.open('../Images/whiteQueenWhiteSquare.png')
            whiteQueenWhiteSquare = whiteQueenWhiteSquare.resize((90, 90), Image.Resampling.LANCZOS)
            self.canvas.whiteQueenWhiteSquareImage = ImageTk.PhotoImage(whiteQueenWhiteSquare,master = self)
            
            whiteQueenBlackSquare = Image.open('../Images/whiteQueenBlackSquare.png')
            whiteQueenBlackSquare = whiteQueenBlackSquare.resize((90, 90), Image.Resampling.LANCZOS)
            self.canvas.whiteQueenBlackSquareImage = ImageTk.PhotoImage(whiteQueenBlackSquare,master = self)
            
            whiteKingWhiteSquare = Image.open('../Images/whiteKingWhiteSquare.png')
            whiteKingWhiteSquare = whiteKingWhiteSquare.resize((90, 90), Image.Resampling.LANCZOS)
            self.canvas.whiteKingWhiteSquareImage = ImageTk.PhotoImage(whiteKingWhiteSquare,master = self)
            
            whiteKingInCheck = Image.open('../Images/WhiteKingInCheck.png')
            whiteKingInCheck = whiteKingInCheck.resize((90, 90), Image.Resampling.LANCZOS)
            self.canvas.whiteKingInCheckImage = ImageTk.PhotoImage(whiteKingInCheck,master = self)
            
            blackKingInCheck = Image.open('../Images/BlackKingInCheck.png')
            blackKingInCheck = blackKingInCheck.resize((90, 90), Image.Resampling.LANCZOS)
            self.canvas.blackKingInCheckImage = ImageTk.PhotoImage(blackKingInCheck,master = self)
        elif platform.system() == 'Linux':
            blackPawnWhiteSquare = Image.open('/mnt/c/Users/Kumodth/Desktop/Programming/Chess Engine/Chess-Engine/Images/BlackPawnWhiteSquare.png')
            blackPawnWhiteSquare = blackPawnWhiteSquare.resize((90, 90), Image.Resampling.LANCZOS)
            self.canvas.blackPawnWhiteSquareImage = ImageTk.PhotoImage(blackPawnWhiteSquare,master = self)
            
            blackPawnBlackSquare = Image.open('/mnt/c/Users/Kumodth/Desktop/Programming/Chess Engine/Chess-Engine/Images/BlackPawnBlackSquare.png')
            blackPawnBlackSquare = blackPawnBlackSquare.resize((90, 90), Image.Resampling.LANCZOS)
            self.canvas.blackPawnBlackSquareImage = ImageTk.PhotoImage(blackPawnBlackSquare,master = self)
            
            blackRookWhiteSquare = Image.open('/mnt/c/Users/Kumodth/Desktop/Programming/Chess Engine/Chess-Engine/Images/BlackRookWhiteSquare.png')
            blackRookWhiteSquare = blackRookWhiteSquare.resize((90, 90), Image.Resampling.LANCZOS)
            self.canvas.blackRookWhiteSquareImage = ImageTk.PhotoImage(blackRookWhiteSquare,master = self)
            
            blackKingWhiteSquare = Image.open('/mnt/c/Users/Kumodth/Desktop/Programming/Chess Engine/Chess-Engine/Images/BlackKingWhiteSquare.png')
            blackKingWhiteSquare = blackKingWhiteSquare.resize((90, 90), Image.Resampling.LANCZOS)
            self.canvas.blackKingWhiteSquareImage = ImageTk.PhotoImage(blackKingWhiteSquare,master = self)
            
            blackKingBlackSquare = Image.open('/mnt/c/Users/Kumodth/Desktop/Programming/Chess Engine/Chess-Engine/Images/BlackKingBlackSquare.png')
            blackKingBlackSquare = blackKingBlackSquare.resize((90, 90), Image.Resampling.LANCZOS)
            self.canvas.blackKingBlackSquareImage = ImageTk.PhotoImage(blackKingBlackSquare,master = self)
            
            blackKnightWhiteSquare = Image.open('/mnt/c/Users/Kumodth/Desktop/Programming/Chess Engine/Chess-Engine/Images/BlackKnightWhiteSquare.png')
            blackKnightWhiteSquare = blackKnightWhiteSquare.resize((90, 90), Image.Resampling.LANCZOS)
            self.canvas.blackKnightWhiteSquareImage = ImageTk.PhotoImage(blackKnightWhiteSquare,master = self)
            
            blackKnightBlackSquare = Image.open('/mnt/c/Users/Kumodth/Desktop/Programming/Chess Engine/Chess-Engine/Images/BlackKnightBlackSquare.png')
            blackKnightBlackSquare = blackKnightBlackSquare.resize((90, 90), Image.Resampling.LANCZOS)
            self.canvas.blackKnightBlackSquareImage = ImageTk.PhotoImage(blackKnightBlackSquare,master = self)
            
            blackRookBlackSquare = Image.open('/mnt/c/Users/Kumodth/Desktop/Programming/Chess Engine/Chess-Engine/Images/BlackRookBlackSquare.png')
            blackRookBlackSquare = blackRookBlackSquare.resize((90, 90), Image.Resampling.LANCZOS)
            self.canvas.blackRookBlackSquareImage = ImageTk.PhotoImage(blackRookBlackSquare,master = self)
            
            blackQueenBlackSquare = Image.open('/mnt/c/Users/Kumodth/Desktop/Programming/Chess Engine/Chess-Engine/Images/BlackQueenBlackSquare.png')
            blackQueenBlackSquare = blackQueenBlackSquare.resize((90, 90), Image.Resampling.LANCZOS)
            self.canvas.blackQueenBlackSquareImage = ImageTk.PhotoImage(blackQueenBlackSquare,master = self)
            
            blackQueenWhiteSquare = Image.open('/mnt/c/Users/Kumodth/Desktop/Programming/Chess Engine/Chess-Engine/Images/BlackQueenWhiteSquare.png')
            blackQueenWhiteSquare = blackQueenWhiteSquare.resize((90, 90), Image.Resampling.LANCZOS)
            self.canvas.blackQueenWhiteSquareImage = ImageTk.PhotoImage(blackQueenWhiteSquare,master = self)
            
            blackBishopWhiteSquare = Image.open('/mnt/c/Users/Kumodth/Desktop/Programming/Chess Engine/Chess-Engine/Images/BlackBishopWhiteSquare.png')
            blackBishopWhiteSquare = blackBishopWhiteSquare.resize((90, 90), Image.Resampling.LANCZOS)
            self.canvas.blackBishopWhiteSquareImage = ImageTk.PhotoImage(blackBishopWhiteSquare,master = self)
              
            blackBishopBlackSquare = Image.open('/mnt/c/Users/Kumodth/Desktop/Programming/Chess Engine/Chess-Engine/Images/BlackBishopBlackSquare.png')
            blackBishopBlackSquare = blackBishopBlackSquare.resize((90, 90), Image.Resampling.LANCZOS)
            self.canvas.blackBishopBlackSquareImage = ImageTk.PhotoImage(blackBishopBlackSquare,master = self) 
            
            whitePawnWhiteSquare = Image.open('/mnt/c/Users/Kumodth/Desktop/Programming/Chess Engine/Chess-Engine/Images/whitePawnWhiteSquare.png')
            whitePawnWhiteSquare = whitePawnWhiteSquare.resize((90, 90), Image.Resampling.LANCZOS)
            self.canvas.whitePawnWhiteSquareImage = ImageTk.PhotoImage(whitePawnWhiteSquare,master = self)

            whitePawnBlackSquare = Image.open('/mnt/c/Users/Kumodth/Desktop/Programming/Chess Engine/Chess-Engine/Images/whitePawnBlackSquare.png')
            whitePawnBlackSquare = whitePawnBlackSquare.resize((90, 90), Image.Resampling.LANCZOS)
            self.canvas.whitePawnBlackSquareImage = ImageTk.PhotoImage(whitePawnBlackSquare,master = self)
            
            whiteRookWhiteSquare = Image.open('/mnt/c/Users/Kumodth/Desktop/Programming/Chess Engine/Chess-Engine/Images/whiteRookWhiteSquare.png')
            whiteRookWhiteSquare = whiteRookWhiteSquare.resize((90, 90), Image.Resampling.LANCZOS)
            self.canvas.whiteRookWhiteSquareImage = ImageTk.PhotoImage(whiteRookWhiteSquare,master = self)
            
            whiteRookBlackSquare = Image.open('/mnt/c/Users/Kumodth/Desktop/Programming/Chess Engine/Chess-Engine/Images/whiteRookBlackSquare.png')
            whiteRookBlackSquare = whiteRookBlackSquare.resize((90, 90), Image.Resampling.LANCZOS)
            self.canvas.whiteRookBlackSquareImage = ImageTk.PhotoImage(whiteRookBlackSquare,master = self)
            
            whiteKnightWhiteSquare = Image.open('/mnt/c/Users/Kumodth/Desktop/Programming/Chess Engine/Chess-Engine/Images/whiteKnightWhiteSquare.png')
            whiteKnightWhiteSquare = whiteKnightWhiteSquare.resize((90, 90), Image.Resampling.LANCZOS)
            self.canvas.whiteKnightWhiteSquareImage = ImageTk.PhotoImage(whiteKnightWhiteSquare,master = self)
            
            whiteKnightBlackSquare = Image.open('/mnt/c/Users/Kumodth/Desktop/Programming/Chess Engine/Chess-Engine/Images/whiteKnightBlackSquare.png')
            whiteKnightBlackSquare = whiteKnightBlackSquare.resize((90, 90), Image.Resampling.LANCZOS)
            self.canvas.whiteKnightBlackSquareImage = ImageTk.PhotoImage(whiteKnightBlackSquare,master = self)
            
            whiteKingBlackSquare = Image.open('/mnt/c/Users/Kumodth/Desktop/Programming/Chess Engine/Chess-Engine/Images/whiteKingBlackSquare.png')
            whiteKingBlackSquare = whiteKingBlackSquare.resize((90, 90), Image.Resampling.LANCZOS)
            self.canvas.whiteKingBlackSquareImage = ImageTk.PhotoImage(whiteKingBlackSquare,master = self)
            
            whiteBishopWhiteSquare = Image.open('/mnt/c/Users/Kumodth/Desktop/Programming/Chess Engine/Chess-Engine/Images/whiteBishopWhiteSquare.png')
            whiteBishopWhiteSquare = whiteBishopWhiteSquare.resize((90, 90), Image.Resampling.LANCZOS)
            self.canvas.whiteBishopWhiteSquareImage = ImageTk.PhotoImage(whiteBishopWhiteSquare,master = self)
            
            whiteBishopBlackSquare = Image.open('/mnt/c/Users/Kumodth/Desktop/Programming/Chess Engine/Chess-Engine/Images/whiteBishopBlackSquare.png')
            whiteBishopBlackSquare = whiteBishopBlackSquare.resize((90, 90), Image.Resampling.LANCZOS)
            self.canvas.whiteBishopBlackSquareImage = ImageTk.PhotoImage(whiteBishopBlackSquare,master = self)
            
            whiteQueenWhiteSquare = Image.open('/mnt/c/Users/Kumodth/Desktop/Programming/Chess Engine/Chess-Engine/Images/whiteQueenWhiteSquare.png')
            whiteQueenWhiteSquare = whiteQueenWhiteSquare.resize((90, 90), Image.Resampling.LANCZOS)
            self.canvas.whiteQueenWhiteSquareImage = ImageTk.PhotoImage(whiteQueenWhiteSquare,master = self)
            
            whiteQueenBlackSquare = Image.open('/mnt/c/Users/Kumodth/Desktop/Programming/Chess Engine/Chess-Engine/Images/whiteQueenBlackSquare.png')
            whiteQueenBlackSquare = whiteQueenBlackSquare.resize((90, 90), Image.Resampling.LANCZOS)
            self.canvas.whiteQueenBlackSquareImage = ImageTk.PhotoImage(whiteQueenBlackSquare,master = self)
            
            whiteKingWhiteSquare = Image.open('/mnt/c/Users/Kumodth/Desktop/Programming/Chess Engine/Chess-Engine/Images/whiteKingWhiteSquare.png')
            whiteKingWhiteSquare = whiteKingWhiteSquare.resize((90, 90), Image.Resampling.LANCZOS)
            self.canvas.whiteKingWhiteSquareImage = ImageTk.PhotoImage(whiteKingWhiteSquare,master = self)
            
            whiteKingInCheck = Image.open('/mnt/c/Users/Kumodth/Desktop/Programming/Chess Engine/Chess-Engine/Images/WhiteKingInCheck.png')
            whiteKingInCheck = whiteKingInCheck.resize((90, 90), Image.Resampling.LANCZOS)
            self.canvas.whiteKingInCheckImage = ImageTk.PhotoImage(whiteKingInCheck,master = self)
            
            blackKingInCheck = Image.open('/mnt/c/Users/Kumodth/Desktop/Programming/Chess Engine/Chess-Engine/Images/BlackKingInCheck.png')
            blackKingInCheck = blackKingInCheck.resize((90, 90), Image.Resampling.LANCZOS)
            self.canvas.blackKingInCheckImage = ImageTk.PhotoImage(blackKingInCheck,master = self)
        
        
        # Create the main board pieces and bind them to their respective coordinates

        for i in range(8):
            
            if i % 2 == 1:
                blackPawn = self.canvas.blackPawnWhiteSquareImage
                whitePawn = self.canvas.whitePawnBlackSquareImage
                index = i
            else:
                blackPawn = self.canvas.blackPawnBlackSquareImage
                whitePawn = self.canvas.whitePawnWhiteSquareImage
                index = i
            blackPawns = self.canvas.create_image(45 + i*90,135,image = blackPawn, anchor = 'center')
            whitePawns = self.canvas.create_image(45 + i*90,585,image = whitePawn, anchor = 'center')
            self.canvas.tag_bind(blackPawns,"<Button-1>", lambda e, i = index + 1, j=7: self.get_location(e,i,7))
            self.canvas.tag_bind(whitePawns,"<Button-1>", lambda e, i = index + 1, j=2: self.get_location(e,i,2))
        
        blackRookWS =self.canvas.create_image(45,45,image = self.canvas.blackRookWhiteSquareImage, anchor = 'center')
        self.canvas.tag_bind(blackRookWS,"<Button-1>", lambda e, i=col+1, j=row+1: self.get_location(e,1,8))    
        
        blackKingWS =self.canvas.create_image(405,45,image = self.canvas.blackKingWhiteSquareImage, anchor = 'center')
        self.canvas.tag_bind(blackKingWS,"<Button-1>", lambda e, i=col+1, j=row+1: self.get_location(e,5,8))  
        
        blackKnightWS = self.canvas.create_image(585,45,image = self.canvas.blackKnightWhiteSquareImage, anchor = 'center')
        self.canvas.tag_bind(blackKnightWS,"<Button-1>", lambda e, i=col+1, j=row+1: self.get_location(e,7,8))
        
        blackKnightBS = self.canvas.create_image(135,45,image = self.canvas.blackKnightBlackSquareImage, anchor = 'center')
        self.canvas.tag_bind(blackKnightBS,"<Button-1>", lambda e, i=col+1, j=row+1: self.get_location(e,2,8))
        
        blackRookBS = self.canvas.create_image(675,45,image = self.canvas.blackRookBlackSquareImage, anchor = 'center')
        self.canvas.tag_bind(blackRookBS,"<Button-1>", lambda e, i=col+1, j=row+1: self.get_location(e,8,8))    
        
        blackQueenBS = self.canvas.create_image(315,45,image = self.canvas.blackQueenBlackSquareImage, anchor = 'center')
        self.canvas.tag_bind(blackQueenBS,"<Button-1>", lambda e, i=col+1, j=row+1: self.get_location(e,4,8))    
        
        blackBishopWS = self.canvas.create_image(225,45,image = self.canvas.blackBishopWhiteSquareImage, anchor = 'center')
        self.canvas.tag_bind(blackBishopWS,"<Button-1>", lambda e, i=col+1, j=row+1: self.get_location(e,3,8))    
        
        blackBishopBS = self.canvas.create_image(495,45,image = self.canvas.blackBishopBlackSquareImage, anchor = 'center')
        self.canvas.tag_bind(blackBishopBS,"<Button-1>", lambda e, i=col+1, j=row+1: self.get_location(e,6,8))
        
        whiteRookWS = self.canvas.create_image(675,675,image = self.canvas.whiteRookWhiteSquareImage, anchor = 'center')
        self.canvas.tag_bind(whiteRookWS,"<Button-1>", lambda e, i=col+1, j=row+1: self.get_location(e,8,1))
        
        whiteRookBS = self.canvas.create_image(45,675,image = self.canvas.whiteRookBlackSquareImage, anchor = 'center')
        self.canvas.tag_bind(whiteRookBS,"<Button-1>", lambda e, i=col+1, j=row+1: self.get_location(e,1,1))
        
        whiteKnightWS = self.canvas.create_image(135,675,image = self.canvas.whiteKnightWhiteSquareImage, anchor = 'center')
        self.canvas.tag_bind(whiteKnightWS,"<Button-1>", lambda e, i=col+1, j=row+1: self.get_location(e,2,1))
        
        whiteKnightBS = self.canvas.create_image(585,675,image = self.canvas.whiteKnightBlackSquareImage, anchor = 'center')
        self.canvas.tag_bind(whiteKnightBS,"<Button-1>", lambda e, i=col+1, j=row+1: self.get_location(e,7,1))
        
        whiteKingBS = self.canvas.create_image(405,675,image = self.canvas.whiteKingBlackSquareImage, anchor = 'center')
        self.canvas.tag_bind(whiteKingBS,"<Button-1>", lambda e, i=col+1, j=row+1: self.get_location(e,5,1))  
        
        whiteBishopWS = self.canvas.create_image(495,675,image = self.canvas.whiteBishopWhiteSquareImage, anchor = 'center')
        self.canvas.tag_bind(whiteBishopWS,"<Button-1>", lambda e, i=col+1, j=row+1: self.get_location(e,6,1))    
        
        whiteBishopBS = self.canvas.create_image(225,675,image = self.canvas.whiteBishopBlackSquareImage, anchor = 'center')
        self.canvas.tag_bind(whiteBishopBS,"<Button-1>", lambda e, i=col+1, j=row+1: self.get_location(e,3,1))
        
        whiteQueenBS = self.canvas.create_image(315,675,image = self.canvas.whiteQueenWhiteSquareImage, anchor = 'center')
        self.canvas.tag_bind(whiteQueenBS,"<Button-1>", lambda e, i=col+1, j=row+1: self.get_location(e,4,1))    
        
        # Sidebar pieces
        
        whiteQueen = self.canvas.create_image(855,45,image = self.canvas.whiteQueenWhiteSquareImage, anchor = 'center')
        self.canvas.tag_bind(whiteQueen,"<Button-1>", lambda e, i="Queen" , j = "White": self.promotion(e,i,j))    
        
        whiteRook = self.canvas.create_image(855,135,image = self.canvas.whiteRookWhiteSquareImage, anchor = 'center')
        self.canvas.tag_bind(whiteRook,"<Button-1>", lambda e, i="Rook", j = "White": self.promotion(e,i,j)) 
        
        whiteBishop = self.canvas.create_image(855,225,image = self.canvas.whiteBishopWhiteSquareImage, anchor = 'center')
        self.canvas.tag_bind(whiteBishop,"<Button-1>", lambda e, i="Bishop", j = "White": self.promotion(e,i,j))    
        
        whiteKnight = self.canvas.create_image(855,315,image = self.canvas.whiteKnightWhiteSquareImage, anchor = 'center')
        self.canvas.tag_bind(whiteKnight,"<Button-1>", lambda e, i="Knight", j = "White": self.promotion(e,i,j))
        
        blackQueen = self.canvas.create_image(855,405,image = self.canvas.blackQueenWhiteSquareImage, anchor = 'center')
        self.canvas.tag_bind(blackQueen,"<Button-1>", lambda e, i="Queen", j = "Black": self.promotion(e,i,j))    
        
        blackRook = self.canvas.create_image(855,495,image = self.canvas.blackRookWhiteSquareImage, anchor = 'center')
        self.canvas.tag_bind(blackRook,"<Button-1>", lambda e, i="Rook", j = "Black": self.promotion(e,i,j)) 
        
        blackBishop = self.canvas.create_image(855,585,image = self.canvas.blackBishopWhiteSquareImage, anchor = 'center')
        self.canvas.tag_bind(blackBishop,"<Button-1>", lambda e, i="Bishop", j = "Black": self.promotion(e,i,j))    
        
        blackKnight = self.canvas.create_image(855,675,image = self.canvas.blackKnightWhiteSquareImage, anchor = 'center')
        self.canvas.tag_bind(blackKnight,"<Button-1>", lambda e, i="Knight", j = "Black": self.promotion(e,i,j))  
    
    def promotion(self,event,pieceChosen,colour):
        print(colour + " " + pieceChosen)
        piece = None
        allow = False
        
        # Check if the promotion flag has been set
        if(self.isPromotion):
            
            # Ensure that the turn and the colour piece selected is correct
            if (self.numMove % 2 == 0):
                if (colour == "Black"):
                    allow = True
                    # Choose the correct image given the square colour of the promotion square
                    if (pieceChosen == "Queen"):
                        self.pawnToBePromoted = Pieces("Queen","Black",self.pawnToBePromoted.xLocation,self.pawnToBePromoted.yLocation)
                        if (self.findSquareColour(self.pawnToBePromoted.xLocation,self.pawnToBePromoted.yLocation) == "White"):   
                            piece = self.canvas.create_image(45 + (self.pawnToBePromoted.xLocation - 1)*90,675,image = self.canvas.blackQueenWhiteSquareImage, anchor = 'center')
                        else:
                            piece = self.canvas.create_image(45 + (self.pawnToBePromoted.xLocation - 1)*90,675,image = self.canvas.blackQueenBlackSquareImage, anchor = 'center')
                    elif(pieceChosen == "Rook"):
                        self.pawnToBePromoted = Pieces("Rook","Black",self.pawnToBePromoted.xLocation,self.pawnToBePromoted.yLocation)
                        if (self.findSquareColour(self.pawnToBePromoted.xLocation,self.pawnToBePromoted.yLocation) == "White"):   
                            piece = self.canvas.create_image(45 + (self.pawnToBePromoted.xLocation - 1)*90,675,image = self.canvas.blackRookWhiteSquareImage, anchor = 'center')
                        else:
                            piece = self.canvas.create_image(45 + (self.pawnToBePromoted.xLocation - 1)*90,675,image = self.canvas.blackRookBlackSquareImage, anchor = 'center')
                    elif(pieceChosen == "Bishop"):
                        self.pawnToBePromoted = Pieces("Bishop","Black",self.pawnToBePromoted.xLocation,self.pawnToBePromoted.yLocation)
                        if (self.findSquareColour(self.pawnToBePromoted.xLocation,self.pawnToBePromoted.yLocation) == "White"):   
                            piece = self.canvas.create_image(45 + (self.pawnToBePromoted.xLocation - 1)*90,675,image = self.canvas.blackBishopWhiteSquareImage, anchor = 'center')
                        else:
                            piece = self.canvas.create_image(45 + (self.pawnToBePromoted.xLocation - 1)*90,675,image = self.canvas.blackBishopBlackSquareImage, anchor = 'center')        
                    elif(pieceChosen == "Knight"):
                        self.pawnToBePromoted = Pieces("Knight","Black",self.pawnToBePromoted.xLocation,self.pawnToBePromoted.yLocation)
                        if (self.findSquareColour(self.pawnToBePromoted.xLocation,self.pawnToBePromoted.yLocation) == "White"):   
                            piece = self.canvas.create_image(45 + (self.pawnToBePromoted.xLocation - 1)*90,675,image = self.canvas.blackKnightWhiteSquareImage, anchor = 'center')
                        else:
                            piece = self.canvas.create_image(45 + (self.pawnToBePromoted.xLocation - 1)*90,675,image = self.canvas.blackKnightBlackSquareImage, anchor = 'center')       
            
            # Ensure that the turn and the colour piece selected is correct
            else:
                # Choose the correct image given the square colour of the promotion square
                if (colour == "White"):
                    allow = True
                    if (pieceChosen == "Queen"):
                        self.pawnToBePromoted = Pieces("Queen","White",self.pawnToBePromoted.xLocation,8)
                        if (self.findSquareColour(self.pawnToBePromoted.xLocation,self.pawnToBePromoted.yLocation) == "White"):   
                            piece = self.canvas.create_image(45 + (self.pawnToBePromoted.xLocation - 1)*90,45,image = self.canvas.whiteQueenWhiteSquareImage, anchor = 'center')
                        else:
                            piece = self.canvas.create_image(45 + (self.pawnToBePromoted.xLocation - 1)*90,45,image = self.canvas.whiteQueenBlackSquareImage, anchor = 'center')
                    elif(pieceChosen == "Rook"):
                        self.pawnToBePromoted = Pieces("Rook","White",self.pawnToBePromoted.xLocation,self.pawnToBePromoted.yLocation)
                        if (self.findSquareColour(self.pawnToBePromoted.xLocation,self.pawnToBePromoted.yLocation) == "White"):   
                            piece = self.canvas.create_image(45 + (self.pawnToBePromoted.xLocation - 1)*90,45,image = self.canvas.whiteRookWhiteSquareImage, anchor = 'center')
                        else:
                            piece = self.canvas.create_image(45 + (self.pawnToBePromoted.xLocation - 1)*90,45,image = self.canvas.whiteRookBlackSquareImage, anchor = 'center')
                    elif(pieceChosen == "Bishop"):
                        self.pawnToBePromoted = Pieces("Bishop","White",self.pawnToBePromoted.xLocation,self.pawnToBePromoted.yLocation)
                        if (self.findSquareColour(self.pawnToBePromoted.xLocation,self.pawnToBePromoted.yLocation) == "White"):   
                            piece = self.canvas.create_image(45 + (self.pawnToBePromoted.xLocation - 1)*90,45,image = self.canvas.whiteBishopWhiteSquareImage, anchor = 'center')
                        else:
                            piece = self.canvas.create_image(45 + (self.pawnToBePromoted.xLocation - 1)*90,45,image = self.canvas.whiteBishopBlackSquareImage, anchor = 'center')        
                    elif(pieceChosen == "Knight"):
                        self.pawnToBePromoted = Pieces("Knight","White",self.pawnToBePromoted.xLocation,self.pawnToBePromoted.yLocation)
                        if (self.findSquareColour(self.pawnToBePromoted.xLocation,self.pawnToBePromoted.yLocation) == "White"):   
                            piece = self.canvas.create_image(45 + (self.pawnToBePromoted.xLocation - 1)*90,45,image = self.canvas.whiteKnightWhiteSquareImage, anchor = 'center')
                        else:
                            piece = self.canvas.create_image(45 + (self.pawnToBePromoted.xLocation - 1)*90,45,image = self.canvas.whiteKnightBlackSquareImage, anchor = 'center') 
            
            # Check if the above conditions were meet
            if (allow):
                
                # Bind the image with its coordinates
                self.canvas.tag_bind(piece,"<Button-1>", lambda e, i=self.pawnToBePromoted.xLocation, j=self.pawnToBePromoted.yLocation: self.get_location(e,i,j))
                self.boardPieces[self.pawnToBePromoted.xLocation - 1][self.pawnToBePromoted.yLocation - 1] = self.pawnToBePromoted
                print(self.pawnToBePromoted.piece)
                print(self.boardPieces[self.pawnToBePromoted.xLocation - 1][self.pawnToBePromoted.yLocation - 1].piece)
                self.isPromotion = False
                
                x = chr(self.PromotionPieceXLocation + 96)
                
                i = chr(self.pawnToBePromoted.xLocation + 96)
                
                if (NNEngine.pgnBoard.turn):
                    NNEngine.pgnBoard.push(chess.Move.from_uci(x+str(7)+i+str(8)+self.pawnToBePromoted.piece[0:1].lower()))
                else:
                    NNEngine.pgnBoard.push(chess.Move.from_uci(x+str(2)+i+str(1)+self.pawnToBePromoted.piece[0:1].lower()))
                
                if (self.numMove % 2 == 0):
                    
                    # Check if the promotion leaves the opponent in check or checkmate
                    if(Rules.isInCheck(chess.WHITE)):
                        if(Rules.isCheckMate(chess.WHITE)):
                            easygui.msgbox("Black Wins!", title="Winner!")
                        kingPiece = next(
                            (y for x in self.boardPieces for y in x if y.piece == "King" and y.colour == "White"), 
                            None
                        )      
                        newPiece = self.canvas.whiteKingInCheckImage
                        self.isCheck = True
                    elif(Rules.isStaleMate(chess.WHITE)):
                        easygui.msgbox("Draw by stalement", title="Draw")
                        
                else:
                    
                    # Check if the promotion leaves the opponent in check or checkmate
                    if(Rules.isInCheck(chess.BLACK)): 
                        if(Rules.isCheckMate(chess.BLACK)):
                            easygui.msgbox("White Wins!", title="Winner!")
                        kingPiece = next(
                            (y for x in self.boardPieces for y in x if y.piece == "King" and y.colour == "Black"), 
                            None
                        )      
                        newPiece = self.canvas.blackKingInCheckImage
                        self.isCheck = True
                    elif(Rules.isStaleMate(chess.BLACK)):
                        easygui.msgbox("Draw by stalement", title="Draw")
                
                # If in check, change the king piece to reflect that
                if(self.isCheck):                 
                    pieces = self.canvas.create_image(45 + (kingPiece.xLocation-1)*90,45 + (8-kingPiece.yLocation)*90,image = newPiece, anchor = 'center')
                    self.canvas.tag_bind(pieces,"<Button-1>", lambda e, x = kingPiece.xLocation, y=kingPiece.yLocation: self.get_location(e,x,y))
                
                # Update the board graphics
                board.update()
                
                    
                # Call the engine to make its move
                if(not(self.isComputerMove)):
                    self.computerThinking = True
                    print ("Move: " + str(self.numMove))
                    print("Position: " + str(NNEngine.evaluateBoard(self)))
                    self.computerThinking = False
                    print(self.PromotionPieceXLocation,str(7),self.pawnToBePromoted.xLocation,str(8),self.pawnToBePromoted.piece[0:1].lower())
                    NNEngine.engineMove(self)
                    
        pass

    # Function to make a move occur graphically
    def get_location(self, event, i, j):
        
        isLegal = False
        isCapture = False
        piece = None
        gameOver = False
        
        if (not(self.isPromotion)):
            
            if (self.move):
                
                # Current item holds the square that wil be moved to
                curItem = self.boardPieces[i-1][j-1] 
                curColour = self.pieceToBeMoved.colour
                
                tempx = self.pieceToBeMoved.xLocation
                tempy = self.pieceToBeMoved.yLocation
                
                # Check if it is legal to move the selected piece to the requested location
                piece,isLegal,isCapture = Rules.isLegalMove(self,curItem,i,j,curColour)
                
                # If the player attempts to capture via en pasent, then the captured item must be assigned explicitly
                if (self.numMove % 2 == 0):
                    if(self.blackSideEnPasent and self.pieceToBeMoved.piece == "Pawn" and i == self.blackSideEnPasentPawnxLocation and isCapture and self.pieceToBeMoved.yLocation == 5):
                        
                        curItem.piece = self.boardPieces[self.blackSideEnPasentPawnxLocation - 1][4].piece
                        curItem.colour = self.boardPieces[self.blackSideEnPasentPawnxLocation - 1][4].colour
                else:
                    if(self.whiteSideEnPasent and self.pieceToBeMoved.piece == "Pawn" and i == self.whiteSideEnPasentPawnxLocation and isCapture and self.pieceToBeMoved.yLocation == 4):
                        curItem.piece = self.boardPieces[self.whiteSideEnPasentPawnxLocation - 1][3].piece
                        curItem.colour = self.boardPieces[self.whiteSideEnPasentPawnxLocation - 1][3].colour
              
                if(isLegal):
                    
                    # Assign variables pertaining to castling
                    # Check if any of the involved pieces have moved
                    if(not(self.kingSideWhiteRookHasMoved)):
                        if (self.pieceToBeMoved.xLocation == 8 and self.pieceToBeMoved.yLocation == 1):
                            self.kingSideWhiteRookHasMoved = True
                    
                    if(not(self.queenSideWhiteRookHasMoved)):
                        if (self.pieceToBeMoved.xLocation == 1 and self.pieceToBeMoved.yLocation == 1):
                            self.queenSideWhiteRookHasMoved = True
                            
                    if(not(self.kingSideBlackRookHasMoved)):
                        if (self.pieceToBeMoved.xLocation == 8 and self.pieceToBeMoved.yLocation == 8):
                            self.kingSideBlackRookHasMoved = True
                    
                    if(not(self.queenSideBlackRookHasMoved)):
                        if (self.pieceToBeMoved.xLocation == 1 and self.pieceToBeMoved.yLocation == 8):
                            self.queenSideBlackRookHasMoved = True
                    
                    if(not(self.whiteKingHasMoved)):
                        if (self.pieceToBeMoved.piece == "King"):
                            if(self.pieceToBeMoved.colour == "White"):
                                self.whiteKingHasMoved = True
                                
                    if(not(self.blackKingHasMoved)):
                        if (self.pieceToBeMoved.piece == "King"):
                            if(self.pieceToBeMoved.colour == "Black"):
                                self.blackKingHasMoved = True
                    
                    # Assign dimensions for the new square
                    x1 = (self.pieceToBeMoved.xLocation - 1) * 90
                    y1 = (8-self.pieceToBeMoved.yLocation) * 90
                    x2 = x1 + 90
                    y2 = y1 + 90
    
                    # Assign the colour for the square
                    if self.findSquareColour(self.pieceToBeMoved.xLocation,self.pieceToBeMoved.yLocation) == "White":
                        colour = "#%02x%02x%02x" % (255,255,255)
                    else:
                        colour = "#%02x%02x%02x" % (167,47,3)
    
                    # Create the frame for the new square and bind it to the move function with its coordinates
                    pieces = self.canvas.create_image(45 + (i-1)*90,45 + (8-j)*90,image = piece, anchor = 'center')
                    self.canvas.tag_bind(pieces,"<Button-1>", lambda e, x = i, y=j: self.get_location(e,x,y))
        
                    # Create the frame for the moving square and bind it to the move function with its coordinates
                    self.board[i-1][j-1] = self.canvas.create_rectangle(x1, y1, x2, y2, fill=colour, tags=f"tile{self.pieceToBeMoved.xLocation}{self.pieceToBeMoved.yLocation}")    
                    self.canvas.tag_bind(f"tile{self.pieceToBeMoved.xLocation}{self.pieceToBeMoved.yLocation}","<Button-1>", lambda e, x=self.pieceToBeMoved.xLocation, y=self.pieceToBeMoved.yLocation: self.get_location(e,x,y))
                    
                    
                    # Set the destination square as the piece to be moved
                    self.boardPieces[i-1][j-1].piece = self.pieceToBeMoved.piece
                    self.boardPieces[i-1][j-1].colour = self.pieceToBeMoved.colour
                    self.boardPieces[i-1][j-1].value = self.pieceToBeMoved.value
                    
                    # Set the square of the piece to be moved as empty
                    self.boardPieces[self.pieceToBeMoved.xLocation - 1][self.pieceToBeMoved.yLocation-1].piece = "Empty"
                    self.boardPieces[self.pieceToBeMoved.xLocation - 1][self.pieceToBeMoved.yLocation-1].colour = "None"
                    self.boardPieces[self.pieceToBeMoved.xLocation - 1][self.pieceToBeMoved.yLocation-1].value = 0
                    
                    
                    # Convert the coordinates to the move string
                    
                    #print(x,y,i,j)
                    x = chr(tempx + 96)
                    y = str(tempy)
                    i = chr(i + 96)
                    j = str(j)
                    NNEngine.pgnBoard.push(chess.Move.from_uci(x+y+i+j))
                    
                    i = ord(i) - 96
                    j = int(j)

                    # If en pasent, make the new square to replace the captured pawn
                    if (self.numMove % 2 == 0):
                        if(self.blackSideEnPasent and self.pieceToBeMoved.piece == "Pawn" and i == self.blackSideEnPasentPawnxLocation and isCapture and self.pieceToBeMoved.yLocation == 5):
                            x1 = (self.blackSideEnPasentPawnxLocation - 1) * 90
                            y1 = (8-5) * 90
                            x2 = x1 + 90
                            y2 = y1 + 90
                            
                            if self.findSquareColour(self.blackSideEnPasentPawnxLocation,5) == "White":
                                colour = "#%02x%02x%02x" % (255,255,255)
                            else:
                                colour = "#%02x%02x%02x" % (167,47,3)
                            
                            self.board[self.blackSideEnPasentPawnxLocation - 1][4] = self.canvas.create_rectangle(x1, y1, x2, y2, fill=colour, tags=f"tile{self.blackSideEnPasentPawnxLocation}{5}")    
                            self.canvas.tag_bind(f"tile{self.blackSideEnPasentPawnxLocation}{5}","<Button-1>", lambda e, x=self.blackSideEnPasentPawnxLocation, y=5: self.get_location(e,x,y))
                            
                            # Set the captured pawn as empty
                            self.boardPieces[self.blackSideEnPasentPawnxLocation - 1][4].piece = "Empty"
                            self.boardPieces[self.blackSideEnPasentPawnxLocation - 1][4].colour = "None"
                            self.boardPieces[self.blackSideEnPasentPawnxLocation - 1][4].value = 0
                    else:
                        if(self.whiteSideEnPasent and self.pieceToBeMoved.piece == "Pawn" and i == self.whiteSideEnPasentPawnxLocation and isCapture and self.pieceToBeMoved.yLocation == 4):
                            x1 = (self.whiteSideEnPasentPawnxLocation - 1) * 90
                            y1 = (8-4) * 90
                            x2 = x1 + 90
                            y2 = y1 + 90
                            
                            if self.findSquareColour(self.whiteSideEnPasentPawnxLocation,4) == "White":
                                colour = "#%02x%02x%02x" % (255,255,255)
                            else:
                                colour = "#%02x%02x%02x" % (167,47,3)
                            
                            self.board[self.whiteSideEnPasentPawnxLocation - 1][3] = self.canvas.create_rectangle(x1, y1, x2, y2, fill=colour, tags=f"tile{self.whiteSideEnPasentPawnxLocation}{4}")    
                            self.canvas.tag_bind(f"tile{self.whiteSideEnPasentPawnxLocation}{5}","<Button-1>", lambda e, x=self.whiteSideEnPasentPawnxLocation, y=4: self.get_location(e,x,y))
                            
                            # Set the captured pawn as empty
                            self.boardPieces[self.whiteSideEnPasentPawnxLocation - 1][3].piece = "Empty"
                            self.boardPieces[self.whiteSideEnPasentPawnxLocation - 1][3].colour = "None"
                            self.boardPieces[self.whiteSideEnPasentPawnxLocation - 1][3].value = 0

                    # If the move is to castle, make the moved rook piece and the empty piece
                    if (self.isCastle):
                        if (i == 7 and j == 1):
                            rookToBeMoved = Pieces("Rook","White",6,1)
                            emptySpace = Pieces("Empty","None",8,1)
                            piece = self.canvas.whiteRookWhiteSquareImage                            
                            colour = "#%02x%02x%02x" % (255,255,255)
                            self.kingSideWhiteRookHasMoved = True
                            self.whiteKingHasMoved = True
                        elif(i == 3 and j == 1):
                            rookToBeMoved = Pieces("Rook","White",4,1)
                            emptySpace = Pieces("Empty","None",1,1)
                            piece = self.canvas.whiteRookWhiteSquareImage
                            colour = "#%02x%02x%02x" % (167,47,3)
                            self.queenSideWhiteRookHasMoved = True
                            self.whiteKingHasMoved = True
                        elif (i == 7 and j == 8):
                            rookToBeMoved = Pieces("Rook","Black",6,8)
                            emptySpace = Pieces("Empty","None",8,8)
                            piece = self.canvas.blackRookBlackSquareImage                            
                            colour = "#%02x%02x%02x" % (167,47,3)
                            self.kingSideBlackRookHasMoved = True
                            self.blackKingHasMoved = True
                        elif(i == 3 and j == 8):
                            rookToBeMoved = Pieces("Rook","Black",4,8)
                            emptySpace = Pieces("Empty","None",1,8)
                            piece = self.canvas.blackRookBlackSquareImage     
                            colour = "#%02x%02x%02x" % (255,255,255)
                            self.queenSideBlackRookHasMoved = True
                            self.blackKingHasMoved = True
                        
                        # Graphically create the rook move when castled
                        pieces = self.canvas.create_image(45 + (rookToBeMoved.xLocation-1)*90,45 + (8-rookToBeMoved.yLocation)*90,image = piece, anchor = 'center')
                        self.canvas.tag_bind(pieces,"<Button-1>", lambda e, x = rookToBeMoved.xLocation, y=rookToBeMoved.yLocation: self.get_location(e,x,y))
                          
                        # Create the empty space
                        x1 = (emptySpace.xLocation - 1) * 90
                        y1 = (8-emptySpace.yLocation) * 90
                        x2 = x1 + 90
                        y2 = y1 + 90
                        
                        self.board[emptySpace.xLocation-1][emptySpace.yLocation-1] = self.canvas.create_rectangle(x1, y1, x2, y2, fill=colour, tags=f"tile{emptySpace.xLocation}{emptySpace.yLocation}")    
                        self.canvas.tag_bind(f"tile{emptySpace.xLocation}{emptySpace.yLocation}","<Button-1>", lambda e, x=emptySpace.xLocation, y=emptySpace.yLocation: self.get_location(e,x,y))

                        # Edit the board with the empty space and moved rook
                        self.boardPieces[emptySpace.xLocation-1][emptySpace.yLocation-1].piece = emptySpace.piece
                        self.boardPieces[emptySpace.xLocation-1][emptySpace.yLocation-1].colour = emptySpace.colour
                        self.boardPieces[emptySpace.xLocation-1][emptySpace.yLocation-1].value = emptySpace.value
                        
                        self.boardPieces[rookToBeMoved.xLocation - 1][rookToBeMoved.yLocation-1].piece = rookToBeMoved.piece
                        self.boardPieces[rookToBeMoved.xLocation - 1][rookToBeMoved.yLocation-1].colour = rookToBeMoved.colour
                        self.boardPieces[rookToBeMoved.xLocation - 1][rookToBeMoved.yLocation-1].value = rookToBeMoved.value
                        self.isCastle = False

                    if(isCapture):
                        self.boardPieces[self.pieceToBeMoved.xLocation - 1][self.pieceToBeMoved.yLocation-1].piece = "Empty"
                        self.boardPieces[self.pieceToBeMoved.xLocation - 1][self.pieceToBeMoved.yLocation-1].colour = "None"
                        self.boardPieces[self.pieceToBeMoved.xLocation - 1][self.pieceToBeMoved.yLocation-1].value = 0

                    # Once out of check, reset the king's image to normal
                    if (self.isCheck):
                        if (self.numMove % 2 == 0):
                            self.isCheck = False
                            kingPiece = next(
                                (y for x in self.boardPieces for y in x if y.piece == "King" and y.colour == "White"), 
                                None
                            )      
                                        
                            if (self.findSquareColour(kingPiece.xLocation,kingPiece.yLocation) =="White"):   
                                piece = self.canvas.whiteKingWhiteSquareImage
                            else:
                                piece = self.canvas.whiteKingBlackSquareImage
                        else:
                            self.isCheck = False
                            kingPiece = next(
                                (y for x in self.boardPieces for y in x if y.piece == "King" and y.colour == "Black"), 
                                None
                            )      
                            if (self.findSquareColour(kingPiece.xLocation,kingPiece.yLocation) =="White"):   
                                piece = self.canvas.blackKingWhiteSquareImage
                            else:
                                piece = self.canvas.blackKingBlackSquareImage
                                            
                        pieces = self.canvas.create_image(45 + (kingPiece.xLocation-1)*90,45 + (8-kingPiece.yLocation)*90,image = piece, anchor = 'center')
                        self.canvas.tag_bind(pieces,"<Button-1>", lambda e, x = kingPiece.xLocation, y=kingPiece.yLocation: self.get_location(e,x,y))               
                                    
                    self.move = False
                    if (self.numMove % 2 == 0):
                        
                        # Set the en pasent variables
                        if(self.whiteSideEnPasent):
                            self.whiteSideEnPasent = False
                        
                        if(self.pieceToBeMoved.piece == "Pawn" and curItem.yLocation == 4 and self.pieceToBeMoved.yLocation == 2):
                            self.whiteSideEnPasent = True
                            self.whiteSideEnPasentPawnxLocation = self.pieceToBeMoved.xLocation
                        
                        # Check if the opposing player has been checkmated, if not, change the king square to red to indicate being in check
                        if(Rules.isInCheck(chess.BLACK)):
                            
                            if(Rules.isCheckMate(chess.BLACK)):
                                easygui.msgbox("White Wins!", title="Winner!")
                                gameOver = True
                            kingPiece = next(
                                (y for x in self.boardPieces for y in x if y.piece == "King" and y.colour == "Black"), 
                                None
                            )      
                            piece = self.canvas.blackKingInCheckImage
                            self.isCheck = True
                                        
                        # Check if the move is a pawn promotion
                        elif(self.pieceToBeMoved.piece == "Pawn" and j == 8):
                            self.isPromotion = True
                            self.pawnToBePromoted = curItem
                            self.PromotionPieceXLocation = tempx
                            NNEngine.pgnBoard.pop()
                            if(self.isComputerMove):
                                self.numMove += 1
                                self.promotion(None,self.pieceChosen,"White")
                                self.numMove -= 1
                                self.isCheck = False
                                
                        # Check if the player has no moves yet is not in check (Stalemate)
                        elif(Rules.isStaleMate(chess.BLACK)):
                            easygui.msgbox("Draw by stalemate", title="Draw")
                            gameOver = True
                    else:                            
                        
                        # Set the en pasent variables
                        if(self.blackSideEnPasent):
                            self.blackSideEnPasent = False
                        
                        if(self.pieceToBeMoved.piece == "Pawn" and curItem.yLocation == 5 and self.pieceToBeMoved.yLocation == 7):
                            self.blackSideEnPasent = True
                            self.blackSideEnPasentPawnxLocation = self.pieceToBeMoved.xLocation
                        
                        # Check if the opposing player has been checkmated, if not, change the king square to red to indicate being in check
                        if(Rules.isInCheck(chess.WHITE)):   
                            if(Rules.isCheckMate(chess.WHITE)):
                                easygui.msgbox("Black Wins!", title="Winner!")
                                gameOver = True
                            kingPiece = next(
                                (y for x in self.boardPieces for y in x if y.piece == "King" and y.colour == "White"), 
                                None
                            )      
                            piece = self.canvas.whiteKingInCheckImage
                            self.isCheck = True
                                        
                        # Check if the move is a pawn promotion                
                        elif(self.pieceToBeMoved.piece == "Pawn" and j == 1):
                            self.isPromotion = True
                            self.pawnToBePromoted = curItem
                            self.PromotionPieceXLocation = tempx
                            NNEngine.pgnBoard.pop()
                            if(self.isComputerMove):
                                self.numMove += 1
                                self.promotion(None,self.pieceChosen,"Black")
                                self.numMove -= 1
                                self.isCheck = False
                        # Check if the player has no moves yet is not in check (Stalemate)
                        elif(Rules.isStaleMate(chess.WHITE)):
                            easygui.msgbox("Draw by stalemate", title="Draw")
                            gameOver = True
                    # Set the piece graphically and bind it to its coordinates
                    if(self.isCheck): 
                        pieces = self.canvas.create_image(45 + (kingPiece.xLocation-1)*90,45 + (8-kingPiece.yLocation)*90,image = piece, anchor = 'center')
                        self.canvas.tag_bind(pieces,"<Button-1>", lambda e, x = kingPiece.xLocation, y=kingPiece.yLocation: self.get_location(e,x,y))
                    
                    self.numMove += 1
                    
                    print ("Moves to ", i,j)
                    print ("Move: " + str(self.numMove))
                    print("Position: " + str(NNEngine.evaluateBoard(self)))
                    print('\n')
                    
                    # Update the graphics
                    board.update()
                    
                    # Call the engine to make a move
                    if(not(self.isPromotion or gameOver)):
                        self.computerThinking = True
                        self.move = False

                        self.computerThinking = False
                        if (not(self.isComputerMove)):
                            NNEngine.engineMove(self)
                        else:
                            self.isComputerMove = False
                    
                self.move = False            
                
                
            else:
                if (self.boardPieces[i-1][j-1].piece == "Empty"):
                    print(self.boardPieces[i-1][j-1].piece)
                else:            
                    
                    # On first click, set the selected piece as the one to be moved
                    # Checks if the correct colour piece is selected for the given turn
                    print(self.boardPieces[i-1][j-1].colour + " " + self.boardPieces[i-1][j-1].piece + " at " + str(i) + " " + str(j))
                    if(self.numMove % 2 == 0 and self.boardPieces[i-1][j-1].colour == "White" or self.numMove % 2 == 1 and self.boardPieces[i-1][j-1].colour == "Black"):
                        
                        self.pieceToBeMoved.piece = self.boardPieces[i-1][j-1].piece
                        self.pieceToBeMoved.colour = self.boardPieces[i-1][j-1].colour
                        self.pieceToBeMoved.xLocation = i
                        self.pieceToBeMoved.yLocation = j
                        self.pieceToBeMoved.value = self.boardPieces[i-1][j-1].value
                        self.move = True
    
    # Function that returns the colour of the square on the board
    def findSquareColour(self,i,j):
        if (i + j) % 2 == 0:
            return "Black"
        else:
            return "White"

def generate_pgn(board):
    """Generates a PGN from the current move stack of the board."""
    
    # Create a new game for the PGN
    game = chess.pgn.Game()

    # Replay the moves in the move stack
    node = game
    for move in board.move_stack:
        node = node.add_variation(move)
    
    # Return the PGN as a string
    return str(game)

if __name__ == "__main__":        
    board = Layout()
    board.drawboard()
    #board.test()
    board.mainloop()         
    print(generate_pgn(NNEngine.pgnBoard))