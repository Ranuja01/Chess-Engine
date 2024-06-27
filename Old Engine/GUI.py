import tkinter as tk
import copy 
import easygui
import random
from timeit import default_timer as timer
from numba import njit

from PIL import ImageTk, Image

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
        #elif(self.piece == "King"):
            #self.value = 100000                      

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
        self.depth = 3
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
        
        #self.get_location(None, 8,8)
        #self.IsAttacking (self.boardPieces[0][0],self.boardPieces[7][7])
    
    def evaluateBoard2(self,colour):
        '''
        USE SLOPE REGRESSION TO CHANGE WEIGHTS TO SATISFY 
        EVALUATION FROM THIS FUNCTION
        '''
        
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
        if (colour == "Black"):
            
            if(self.whiteHasCastled):
                sum -= 500
            if(self.blackHasCastled):
                sum += 500
            oppositeColour = "White"
        else:
            
            if(self.blackHasCastled):
                sum -= 500
            if(self.whiteHasCastled):
                sum += 500
            oppositeColour = "Black"
            
        for item in self.boardPieces:
            for square in item:
                if(square.piece == "King"):
                    if(square.colour == "Black"):
                        blackKingPiece = square
                    else:
                        whiteKingPiece = square
            
        if(self.isInCheck(colour)):
            sum -= 150
            if(self.isCheckMate(colour)):
                sum = -9999999999999
        else:
            if(colour == "Black"):
                if(not(self.blackHasCastled)):
                    if(not(blackKingPiece.xLocation == 5 and blackKingPiece.yLocation == 8 )):
                        sum -= 500
            else:
                if(not(self.whiteHasCastled)):
                    if(not(whiteKingPiece.xLocation == 5 and whiteKingPiece.yLocation == 1 )):
                        sum -= 500
            
        if(self.isInCheck(oppositeColour)):
            sum += 150
            if(self.isCheckMate(oppositeColour)):
                sum = 9999999999999
        elif(self.numMove >= 30):
            if(self.isCheckMate(oppositeColour)):
                #print("ASASDASD")
                sum = -9999999999999
        
        
        for item in self.boardPieces:
            for square in item:
                if(square.colour == colour):
                    
                    sum += square.value
                    if(not(square.piece == "King")):
                        
                        if(not(square.piece == "Rook" and activePlacementLayer == placementLayer)):
                        
                            if(colour == "White"):
                                sum += activePlacementLayer[0][square.xLocation - 1][square.yLocation - 1]
                            else:
                                sum += activePlacementLayer[1][square.xLocation - 1][square.yLocation - 1]
 
                        if(sum >= 13200):
                            activeLayer = layer
                            activePlacementLayer = placementLayer
                        elif(self.numMove >= 18):
                            if(square.piece == "Pawn"):
                                if(colour == "White"):
                                    sum += square.yLocation * 25
                                else:
                                    sum += (9 - square.yLocation) * 25
                                
        
        return sum
    
    def evaluateBoard3(this,colour):


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
        
        
        
        if(this.move >= 18):
            activeLayer = layer2
            activePlacementLayer = layer2
        else:
            activeLayer = layer
            activePlacementLayer = placementLayer
        sum = 0
        if (colour == "Black"):
            
            if(this.whiteHasCastled):
                sum -= 500
            if(this.blackHasCastled):
                sum += 500
            oppositeColour = "White"
        else:
            
            if(this.blackHasCastled):
                sum -= 500
            if(this.whiteHasCastled):
                sum += 500
            oppositeColour = "Black"
            
        for item in this.boardPieces:
            for square in item:
                if(square.piece == "King"):
                    if(square.colour == "Black"):
                        blackKingPiece = square
                    else:
                        whiteKingPiece = square
            
        if(this.isInCheck(colour)):
            sum -= 150
            if(this.isCheckMate(colour)):
                sum = -99999995
        else:
            if(colour == "Black"):
                if(not(this.blackHasCastled)):
                    if(not(blackKingPiece.xLocation == 5 and blackKingPiece.yLocation == 8 )):
                        sum -= 500
            else:
                if(not(this.whiteHasCastled)):
                    if(not(whiteKingPiece.xLocation == 5 and whiteKingPiece.yLocation == 1 )):
                        sum -= 500
            
        if(this.isInCheck(oppositeColour)):
            sum += 150
            if(this.isCheckMate(oppositeColour)):
                sum = 9999999999999
        elif(this.numMove >= 30):
            if(this.isCheckMate(oppositeColour)):
                #print("ASASDASD")
                sum = -99999994
        
        
        for item in this.boardPieces:
            for square in item:
                if(square.colour == colour):
                    
                    sum += square.value
                    if(not(square.piece == "King")):
                        
                        if(not(square.piece == "Rook" and activePlacementLayer == placementLayer)):
                        
                            if(colour == "White"):
                                sum += activePlacementLayer[0][square.xLocation - 1][square.yLocation - 1]
                            else:
                                sum += activePlacementLayer[1][square.xLocation - 1][square.yLocation - 1]
        
                        if(sum >= 13200):
                            activeLayer = layer
                            activePlacementLayer = placementLayer
                        elif(this.numMove >= 18):
                            if(square.piece == "Pawn"):
                                if(colour == "White"):
                                    sum += square.yLocation * 25
                                else:
                                    sum += (9 - square.yLocation) * 25
                                
          
        for x in this.boardPieces:
            for y in x:
                if (y.colour == colour and not(y.piece == "Empty") and not(y.piece == "King")):
                    moves = []
                    this.moveAppender(moves,y,colour)
                    
                    for item in moves:
                        this.pieceToBeMoved = this.boardPieces[y.xLocation - 1][y.yLocation - 1]
                        isLegal,isCapture = this.isLegalMove(this.boardPieces[item.xLocation - 1][item.yLocation - 1],item.xLocation,item.yLocation,colour)
                        if (isLegal):
                            if (this.IsAttacking(y,this.boardPieces[item.xLocation - 1][item.yLocation - 1])):
                                
                                if(y.piece == "Pawn"):
                                    multiplier = 9
                                elif(y.piece == "Knight"):
                                    multiplier = 5 
                                elif(y.piece == "Bishop"):
                                    multiplier = 5
                                elif(y.piece == "Rook"):
                                    multiplier = 1   
                                elif(y.piece == "Queen"):
                                    if(colour == "White"):
                                        sum += activeLayer[0][item.xLocation - 1][item.yLocation - 1] // 3
                                    else:
                                        sum += activeLayer[1][item.xLocation - 1][item.yLocation - 1] // 3
                                    multiplier = 1   
                                
                                if(colour == "White"):
                                    sum += activeLayer[0][item.xLocation - 1][item.yLocation - 1] * multiplier
                                else:
                                    sum += activeLayer[1][item.xLocation - 1][item.yLocation - 1] * multiplier
        return sum  
        
    def evaluateBoard(self):
        sum = 0

        #print ('\n')
        if(self.blackHasCastled):
            sum += 1500
        if(self.whiteHasCastled):
            sum -= 1500

        if(self.isInCheck("White")):
            if(self.isCheckMate("White")):
                print("AAAAA")
                sum = 10000000
        elif(self.isInCheck("Black")):
            if(self.isCheckMate("Black")):
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
                    #if (not(self.boardPieces[6][6].piece == "Pawn")):
                        #print(square.piece,square.colour,square.xLocation,square.yLocation, sum)
                        #print(self.boardPieces[6][6].piece,self.boardPieces[6][6].colour,self.boardPieces[6][6].xLocation,self.boardPieces[6][6].yLocation, sum)    
                        
                    if(square.xLocation == 4 and square.yLocation == 4 or square.xLocation == 4 and square.yLocation == 5 or square.xLocation == 5 and square.yLocation == 4 or square.xLocation == 5 and square.yLocation == 5):
                        #print("AAAAA", square.piece,square.colour,square.xLocation,square.yLocation, sum)
                        if (square.colour == "Black"):
                            sum += 150
                        elif (square.colour == "White"):
                            sum -= 150
        
        return sum   
   
    
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
                
                if (self.numMove % 2 == 1):
                    
                    # Check if the promotion leaves the opponent in check or checkmate
                    if(self.isInCheck("White")):
                        if(self.isCheckMate("White")):
                            easygui.msgbox("Black Wins!", title="Winner!")
                        kingPiece = None
                        for x in self.boardPieces:
                            for y in x:
                                if (y.piece == "King" and y.colour == "White"):
                                    kingPiece = y
                                    newPiece = self.canvas.whiteKingInCheckImage
                                    self.isCheck = True
                    elif(self.isCheckMate("White")):
                        easygui.msgbox("Draw by stalement", title="Draw")
                        
                else:
                    
                    # Check if the promotion leaves the opponent in check or checkmate
                    if(self.isInCheck("Black")): 
                        if(self.isCheckMate("Black")):
                            easygui.msgbox("White Wins!", title="Winner!")
                        kingPiece = None
                        for x in self.boardPieces:
                            for y in x:
                                if (y.piece == "King" and y.colour == "Black"):
                                    kingPiece = y
                                    newPiece = self.canvas.blackKingInCheckImage
                                    self.isCheck = True
                    elif(self.isCheckMate("Black")):
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
                    print("Position: " + str(self.evaluateBoard()))
                    self.computerThinking = False
                    self.engineMove("Black")
                    
        pass
   
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
                        if(j == 3 and self.boardPieces[i-1][2].piece == "Empty"):
                            isLegal = True
                        elif(j == 4 and self.boardPieces[i-1][2].piece == "Empty" and self.boardPieces[i-1][3].piece == "Empty"):
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
                         if(j == 6 and self.boardPieces[i-1][5].piece == "Empty"):
                            isLegal = True
                         elif(j == 5 and self.boardPieces[i-1][5].piece == "Empty" and self.boardPieces[i-1][4].piece == "Empty"):
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
                
                if(not(self.boardPieces[i + square*xMultiplier - 1][j + square*yMultiplier - 1].piece == "Empty")):
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
                    if(not(self.boardPieces[i-1][j + square*yMultiplier - 1].piece == "Empty")):
                        isLegal = False
            # Horizontal moves
            else:
                
                # If the defending piece is ahead of the attacking, the check must be done in the other direction
                if(i > self.pieceToBeMoved.xLocation):
                    xMultiplier = -1
            
                # Check if there is an impeding piece in the way, if so the attacker is not attacking the defending piece
                for square in range(1,abs(i - self.pieceToBeMoved.xLocation)):
                    if(not(self.boardPieces[i+ square*xMultiplier - 1][j-1].piece == "Empty")):
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
                
                if(not(self.boardPieces[i + square*xMultiplier - 1][j + square*yMultiplier - 1].piece == "Empty")):
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
                    if(not(self.boardPieces[i-1][j + square*yMultiplier - 1].piece == "Empty")):
                        isLegal = False
            # Horizontal moves
            else:
                
                # If the defending piece is ahead of the attacking, the check must be done in the other direction
                if(i > self.pieceToBeMoved.xLocation):
                    xMultiplier = -1
            
                # Check if there is an impeding piece in the way, if so the attacker is not attacking the defending piece
                for square in range(1,abs(i - self.pieceToBeMoved.xLocation)):
                    if(not(self.boardPieces[i+ square*xMultiplier - 1][j-1].piece == "Empty")):
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
                        if (self.boardPieces[5][0].piece == "Empty" and self.boardPieces[6][0].piece == "Empty" and self.boardPieces[7][0].piece == "Rook" and self.boardPieces[7][0].colour == "White"):
                            self.isCastle = True
                            
                            # Check if any opposing piece is attacking the castling path
                            for x in self.boardPieces:
                                for y in x:
                                    if (y.colour == "Black"):
                                        if(self.IsAttacking(y,self.boardPieces[4][0]) or self.IsAttacking(y,self.boardPieces[5][0]) or self.IsAttacking(y,self.boardPieces[6][0])):
                                             self.isCastle = False
                # White queenside castle                             
                if (i == 3 and j == 1):
                    
                    # Check that neither of the involved pieces have previously moved
                    if(not(self.whiteKingHasMoved) and not(self.queenSideWhiteRookHasMoved)):
                        
                        # Check if there are no pieces obstructing castling
                        if (self.boardPieces[1][0].piece == "Empty" and self.boardPieces[2][0].piece == "Empty" and self.boardPieces[3][0].piece == "Empty" and self.boardPieces[0][0].piece == "Rook" and self.boardPieces[0][0].colour == "White"):
                            self.isCastle = True
                            
                            # Check if any opposing piece is attacking the castling path
                            for x in self.boardPieces:
                                for y in x:
                                    if (y.colour == "Black"):
                                        if(self.IsAttacking(y,self.boardPieces[4][0]) or self.IsAttacking(y,self.boardPieces[1][0]) or self.IsAttacking(y,self.boardPieces[2][0]) or self.IsAttacking(y,self.boardPieces[3][0])):
                                             self.isCastle = False
            else:
                
                # Black kingside castle
                if (i == 7 and j == 8):
                    
                    # Check that neither of the involved pieces have previously moved
                    if(not(self.blackKingHasMoved) and not(self.kingSideBlackRookHasMoved)):
                        
                        # Check if there are no pieces obstructing castling
                        if (self.boardPieces[5][7].piece == "Empty" and self.boardPieces[6][7].piece == "Empty" and self.boardPieces[7][7].piece == "Rook" and self.boardPieces[7][7].colour == "Black"):
                            self.isCastle = True
                            
                            # Check if any opposing piece is attacking the castling path
                            for x in self.boardPieces:
                                for y in x:
                                    if (y.colour == "White"):
                                        if(self.IsAttacking(y,self.boardPieces[4][7]) or self.IsAttacking(y,self.boardPieces[5][7]) or self.IsAttacking(y,self.boardPieces[6][7])):
                                             self.isCastle = False
                # Black queenside castle                             
                if (i == 3 and j == 8):
                    
                    # Check that neither of the involved pieces have previously moved
                    if(not(self.blackKingHasMoved) and not(self.queenSideBlackRookHasMoved)):
                        
                        # Check if there are no pieces obstructing castling
                        if (self.boardPieces[1][7].piece == "Empty" and self.boardPieces[2][7].piece == "Empty" and self.boardPieces[3][7].piece == "Empty" and self.boardPieces[0][7].piece == "Rook" and self.boardPieces[0][7].colour == "Black"):
                            self.isCastle = True
                            
                            # Check if any opposing piece is attacking the castling path
                            for x in self.boardPieces:
                                for y in x:
                                    if (y.colour == "White"):
                                        if(self.IsAttacking(y,self.boardPieces[4][7]) or self.IsAttacking(y,self.boardPieces[1][7]) or self.IsAttacking(y,self.boardPieces[2][7]) or self.IsAttacking(y,self.boardPieces[3][7])):
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
                piece,isLegal,isCapture = self.isLegalPawnMove(curItem,i,j,colour)     
            elif(self.pieceToBeMoved.piece == "Knight"):
                piece,isLegal,isCapture = self.isLegalKnightMove(curItem,i,j,colour) 
            elif(self.pieceToBeMoved.piece == "Bishop"):
                piece,isLegal,isCapture = self.isLegalBishopMove(curItem,i,j,colour) 
            elif(self.pieceToBeMoved.piece == "Rook"):
                piece,isLegal,isCapture = self.isLegalRookMove(curItem,i,j,colour)  
            elif(self.pieceToBeMoved.piece == "Queen"):
                piece,isLegal,isCapture = self.isLegalQueenMove(curItem,i,j,colour)                     
            elif(self.pieceToBeMoved.piece == "King"):
                piece,isLegal,isCapture = self.isLegalKingMove(curItem,i,j,colour)
        
            if (isLegal):

                # Make the location square hold the moving piece
                self.boardPieces[i-1][j-1].piece = self.pieceToBeMoved.piece
                self.boardPieces[i-1][j-1].colour = self.pieceToBeMoved.colour
                
                # Empty the previous square
                self.boardPieces[self.pieceToBeMoved.xLocation - 1][self.pieceToBeMoved.yLocation-1].piece = "Empty"
                self.boardPieces[self.pieceToBeMoved.xLocation - 1][self.pieceToBeMoved.yLocation-1].colour = "None"
                
                # Check if making this move puts the current player under check
                # If so, the move is not legal
                if (self.numMove % 2 == 0):
                    if(self.isInCheck("White")):
                        isLegal = False
                else:
                    if(self.isInCheck("Black")):    
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

    # Function to make a move occur graphically
    def get_location(self, event, i, j):
        
        isLegal = False
        isCapture = False
        piece = None
        
        if (not(self.isPromotion)):
            
            if (self.move):
                
                # Current item holds the square that wil be moved to
                curItem = self.boardPieces[i-1][j-1] 
                curColour = self.pieceToBeMoved.colour
                
                # Check if it is legal to move the selected piece to the requested location
                piece,isLegal,isCapture = self.isLegalMove(curItem,i,j,curColour)
                
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
                                for x in self.boardPieces:
                                    for y in x:
                                        if (y.piece == "King" and y.colour == "White"):
                                            kingPiece = y
                                            
                                            if (self.findSquareColour(kingPiece.xLocation,kingPiece.yLocation) =="White"):   
                                                piece = self.canvas.whiteKingWhiteSquareImage
                                            else:
                                                piece = self.canvas.whiteKingBlackSquareImage
                            else:
                                self.isCheck = False
                                for x in self.boardPieces:
                                    for y in x:
                                        if (y.piece == "King" and y.colour == "Black"):
                                            kingPiece = y
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
                            if(self.isInCheck("Black")):
                                
                                if(self.isCheckMate("Black")):
                                    easygui.msgbox("White Wins!", title="Winner!")
                                
                                kingPiece = None
                                for x in self.boardPieces:
                                    for y in x:
                                        if (y.piece == "King" and y.colour == "Black"):
                                            kingPiece = y
                                            piece = self.canvas.blackKingInCheckImage
                                            self.isCheck = True
                                            
                            # Check if the move is a pawn promotion
                            elif(self.pieceToBeMoved.piece == "Pawn" and j == 8):
                                self.isPromotion = True
                                self.pawnToBePromoted = curItem
                                if(self.isComputerMove):
                                    self.numMove += 1
                                    self.promotion(None,self.pieceChosen,"White")
                                    self.numMove -= 1
                                    self.isCheck = False
                                    
                            # Check if the player has no moves yet is not in check (Stalemate)
                            elif(self.isCheckMate("Black")):
                                easygui.msgbox("Draw by stalemate", title="Draw")
                              
                        else:                            
                            
                            # Set the en pasent variables
                            if(self.blackSideEnPasent):
                                self.blackSideEnPasent = False
                            
                            if(self.pieceToBeMoved.piece == "Pawn" and curItem.yLocation == 5 and self.pieceToBeMoved.yLocation == 7):
                                self.blackSideEnPasent = True
                                self.blackSideEnPasentPawnxLocation = self.pieceToBeMoved.xLocation
                            
                            # Check if the opposing player has been checkmated, if not, change the king square to red to indicate being in check
                            if(self.isInCheck("White")):   
                                if(self.isCheckMate("White")):
                                    easygui.msgbox("Black Wins!", title="Winner!")
                                    
                                kingPiece = None
                                for x in self.boardPieces:
                                    for y in x:
                                        if (y.piece == "King" and y.colour == "White"):
                                            kingPiece = y
                                            piece = self.canvas.whiteKingInCheckImage
                                            self.isCheck = True
                                            
                            # Check if the move is a pawn promotion                
                            elif(self.pieceToBeMoved.piece == "Pawn" and j == 1):
                                self.isPromotion = True
                                self.pawnToBePromoted = curItem
                                if(self.isComputerMove):
                                    self.numMove += 1
                                    self.promotion(None,self.pieceChosen,"Black")
                                    self.numMove -= 1
                                    self.isCheck = False
                            # Check if the player has no moves yet is not in check (Stalemate)
                            elif(self.isCheckMate("White")):
                                easygui.msgbox("Draw by stalemate", title="Draw")
                        
                        # Set the piece graphically and bind it to its coordinates
                        if(self.isCheck): 
                            pieces = self.canvas.create_image(45 + (kingPiece.xLocation-1)*90,45 + (8-kingPiece.yLocation)*90,image = piece, anchor = 'center')
                            self.canvas.tag_bind(pieces,"<Button-1>", lambda e, x = kingPiece.xLocation, y=kingPiece.yLocation: self.get_location(e,x,y))
                        
                        self.numMove += 1
                        
                        print ("Moves to ", i,j)
                        print ("Move: " + str(self.numMove))
                        print("Position: " + str(self.evaluateBoard()))
                        print('\n')
                        
                        # Update the graphics
                        board.update()
                        
                        # Call the engine to make a move
                        if(not(self.isPromotion)):
                            self.computerThinking = True
                            self.move = False

                            self.computerThinking = False
                            if (not(self.isComputerMove)):
                                self.engineMove("Black")
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
    
    # Function that allows the engine to make a move
    def engineMove(self,colour):
        t0= timer()
    
        # Set variable such that certain features know that an actual move is not being attempted
        self.computerThinking = True
        
        # Call the alpha beta algorithm to make a move decision
        currentItem,pieceToBeMoved,val,self.pieceChosen = self.alphaBeta(0,self.depth,"Black")
        self.computerThinking = False
        
        # If the algorithm does not select a move, that suggests they are all equally bad and therefore the machine resigns
        if(not(pieceToBeMoved == None or pieceToBeMoved.piece == "Empty") ):
            
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
            
            self.move = True
            self.get_location(None, currentItem.xLocation, currentItem.yLocation)

        else:
            
            print ("Position: " + str(val))
            print ("Number of Iterations: " + str(self.count))
            easygui.msgbox("Black Resigns", title="Winner!")
    
    # Function that returns the colour of the square on the board
    def findSquareColour(self,i,j):
        if (i + j) % 2 == 0:
            return "Black"
        else:
            return "White"

    # Function to check if the given side is in check        
    def isInCheck(self,colour):
        kingPiece = None
        
        # Find the king piece
        for x in self.boardPieces:
            for y in x:
                if (y.piece == "King" and y.colour == colour):
                    kingPiece = y
        
        # Check if any piece is attacking the king
        for x in self.boardPieces:
            for y in x:
                if (not(y.colour == colour) and not(y.piece == "Empty")):

                    if(self.IsAttacking(y,kingPiece)):
                        return True
        return False
        
        
    # Function to check if a piece if attacking another piece
    def IsAttacking (self,attacker,defender):
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
                    if(not(self.boardPieces[defender.xLocation + square*xMultiplier - 1][defender.yLocation + square*yMultiplier - 1].piece == "Empty")):
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
                        if(not(self.boardPieces[defender.xLocation-1][defender.yLocation + square*yMultiplier - 1].piece == "Empty")):
                            return False
                
                # Horizontal moves   
                else:
                    
                    # If the defending piece is ahead of the attacking, the check must be done in the other direction
                    if(defender.xLocation > attacker.xLocation):
                        xMultiplier = -1
                
                    # Check if there is an impeding piece in the way, if so the attacker is not attacking the defending piece 
                    for square in range(1,abs(defender.xLocation - attacker.xLocation)):
                        if(not(self.boardPieces[defender.xLocation + square*xMultiplier - 1][defender.yLocation-1].piece == "Empty")):
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
                    if(not(self.boardPieces[defender.xLocation + square*xMultiplier - 1][defender.yLocation + square*yMultiplier - 1].piece == "Empty")):
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
                        if(not(self.boardPieces[defender.xLocation-1][defender.yLocation + square*yMultiplier - 1].piece == "Empty")):
                            return False           
                
                # Horizontal moves
                else:
                    # If the defending piece is ahead of the attacking, the check must be done in the other direction
                    if(defender.xLocation > attacker.xLocation):
                        xMultiplier = -1
                    
                    # Check if there is an impeding piece in the way, if so the attacker is not attacking the defending piece
                    for square in range(1,abs(defender.xLocation - attacker.xLocation)):
                        if(not(self.boardPieces[defender.xLocation + square*xMultiplier - 1][defender.yLocation-1].piece == "Empty")):
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
                self.boardPieces[defender.xLocation - 1][defender.yLocation - 1].piece = attacker.piece
                self.boardPieces[defender.xLocation - 1][defender.yLocation - 1].colour = attacker.colour
                
                self.boardPieces[attacker.xLocation - 1][attacker.yLocation - 1].piece = "Empty"
                self.boardPieces[attacker.xLocation - 1][attacker.yLocation - 1].colour = "None"
                 
                if (self.numMove % 2 == 0):
                     
                    if(self.isInCheck("White")):
                        self.boardPieces = copy.deepcopy(boardCopy)
                        return False
                else:
                    if(self.isInCheck("Black")):  
                         self.boardPieces = copy.deepcopy(boardCopy) 
                         return False
                 
                    
                self.boardPieces = copy.deepcopy(boardCopy)  
            return True  
        
        return False 
    
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
            return self.evaluateBoard()
        
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
        castleFlag = False
        
        # Create a pre-move copy of the board
        boardCopy = copy.deepcopy(self.boardPieces)
        
        # Loop through all pieces on the board
        for x in self.boardPieces:
            for y in x:
                
                # Check if the chosen piece is of the active player
                if (y.colour == evalColour):
                    
                    # Acquire list of possible moves for the given piece
                    moves = []
                    self.moveAppender(moves,y,evalColour)

                    # Loop through list of moves
                    for item in moves:
                        
                        # Set the piece to be moved as the current piece                        
                        self.pieceToBeMoved.piece = self.boardPieces[y.xLocation - 1][y.yLocation - 1].piece
                        self.pieceToBeMoved.colour = self.boardPieces[y.xLocation - 1][y.yLocation - 1].colour
                        self.pieceToBeMoved.value = self.boardPieces[y.xLocation - 1][y.yLocation - 1].value
                        self.pieceToBeMoved.xLocation = y.xLocation
                        self.pieceToBeMoved.yLocation = y.yLocation
                        
                        # Check if moving this piece to each move's location is legal
                        isLegal,isCapture = self.isLegalMove(self.boardPieces[item.xLocation - 1][item.yLocation - 1],item.xLocation,item.yLocation,evalColour)

                        if(isLegal):
                            
                            if(not(item.piece == y.piece)):
                                promotion = True
                            
                            self.boardPieces[item.xLocation - 1][item.yLocation - 1].piece = item.piece
                            self.boardPieces[item.xLocation - 1][item.yLocation - 1].colour = item.colour
                            self.boardPieces[item.xLocation - 1][item.yLocation - 1].value = item.value
                            
                            self.boardPieces[y.xLocation - 1][y.yLocation - 1].piece = "Empty"
                            self.boardPieces[y.xLocation - 1][y.yLocation - 1].colour = "None"
                            self.boardPieces[y.xLocation - 1][y.yLocation - 1].value = 0
                            
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
                                if(self.pieceToBeMoved.piece == "Pawn" and item.yLocation == 5 and y.yLocation == 7):
                                    self.blackSideEnPasent = True
                                    self.blackSideEnPasentPawnxLocation = y.xLocation
                                    pass
                                
                                # Remove the pawn being captured if done through en pasent
                                if(self.whiteSideEnPasent and self.pieceToBeMoved.piece == "Pawn" and item.xLocation == self.whiteSideEnPasentPawnxLocation and isCapture and y.yLocation == 4):
                                    self.boardPieces[self.whiteSideEnPasentPawnxLocation - 1][3].piece = "Empty"
                                    self.boardPieces[self.whiteSideEnPasentPawnxLocation - 1][3].colour = "None"
                                    self.boardPieces[self.whiteSideEnPasentPawnxLocation - 1][3].value = 0
                                                    
                            else:
                                
                                # If the piece was set in a previous iteration as in en pasent position, it is no longer by the next move
                                if(self.whiteSideEnPasent):
                                    self.whiteSideEnPasent = False
                                
                                # Set if the pawn is in en pasent position
                                if(self.pieceToBeMoved.piece == "Pawn" and item.yLocation == 4 and y.yLocation == 2):
                                    self.whiteSideEnPasent = True
                                    self.whiteSideEnPasentPawnxLocation = y.xLocation  
                                    pass 
                                
                                # Remove the pawn being captured if done through en pasent
                                if(self.blackSideEnPasent and self.pieceToBeMoved.piece == "Pawn" and item.xLocation == self.blackSideEnPasentPawnxLocation and isCapture and y.yLocation == 5):
                                    self.boardPieces[self.blackSideEnPasentPawnxLocation - 1][4].piece = "Empty"
                                    self.boardPieces[self.blackSideEnPasentPawnxLocation - 1][4].colour = "None"
                                    self.boardPieces[self.blackSideEnPasentPawnxLocation - 1][4].value = 0
                            
                            
                            # Increment the move number to simulate a move being made
                            self.numMove += 1
                            # Call the minimizer to make the next move
                            score = self.minimizer(curDepth + 1,depthLimit,oppositeColour,alpha, beta)
                            self.numMove -= 1
                            
                            
                            if (self.numMove == 35):
                                with open('Unfiltered_Full.txt', 'a') as file:
                                    file.write("MIN CHOSEN: {}, {}, {}, {}, {}\n".format(score, item.piece, item.colour, item.xLocation, item.yLocation))
                            '''
                            if (self.numMove == 11):
                                print ("MIN CHOSEN: ", score, item.piece, item.colour, item.xLocation, item.yLocation)
                                print (score)  
                            '''
                            
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
                                pieceToBeMoved.piece = self.boardPieces[y.xLocation - 1][y.yLocation - 1].piece
                                pieceToBeMoved.colour = self.boardPieces[y.xLocation - 1][y.yLocation - 1].colour
                                pieceToBeMoved.xLocation = y.xLocation 
                                pieceToBeMoved.yLocation = y.yLocation
                                pieceToBeMoved.value = self.boardPieces[y.xLocation - 1][y.yLocation - 1].value
                            
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
            return self.evaluateBoard()
        
        isLegal = False
        isCapture = False
        highestScore = -99999999
        castleFlag = False
        
        # Create a pre-move copy of the board
        boardCopy = copy.deepcopy(self.boardPieces)
        
        # Loop through all pieces on the board
        for x in self.boardPieces:
            for y in x:
                
                # Check if the chosen piece is of the active player
                if (y.colour == evalColour):
                    
                    # Acquire list of possible moves for the given piece
                    moves = []
                    self.moveAppender(moves,y,evalColour)

                    # Loop through list of moves
                    for item in moves:
                        
                        # Set the piece to be moved as the current piece                        
                        self.pieceToBeMoved.piece = self.boardPieces[y.xLocation - 1][y.yLocation - 1].piece
                        self.pieceToBeMoved.colour = self.boardPieces[y.xLocation - 1][y.yLocation - 1].colour
                        self.pieceToBeMoved.value = self.boardPieces[y.xLocation - 1][y.yLocation - 1].value
                        self.pieceToBeMoved.xLocation = y.xLocation
                        self.pieceToBeMoved.yLocation = y.yLocation
                        
                        # Check if moving this piece to each move's location is legal
                        isLegal,isCapture = self.isLegalMove(self.boardPieces[item.xLocation - 1][item.yLocation - 1],item.xLocation,item.yLocation,evalColour)

                        if(isLegal):
                           
                            # If legal, make the move
                            self.boardPieces[item.xLocation - 1][item.yLocation - 1].piece = item.piece
                            self.boardPieces[item.xLocation - 1][item.yLocation - 1].colour = item.colour
                            self.boardPieces[item.xLocation - 1][item.yLocation - 1].value = item.value
                            
                            # Replace starting position with empty square
                            self.boardPieces[y.xLocation - 1][y.yLocation - 1].piece = "Empty"
                            self.boardPieces[y.xLocation - 1][y.yLocation - 1].colour = "None"
                            self.boardPieces[y.xLocation - 1][y.yLocation - 1].value = 0
                            
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
                                if(self.pieceToBeMoved.piece == "Pawn" and item.yLocation == 5 and y.yLocation == 7):
                                    self.blackSideEnPasent = True
                                    self.blackSideEnPasentPawnxLocation = y.xLocation
                                    pass
                                
                                # Remove the pawn being captured if done through en pasent
                                if(self.whiteSideEnPasent and self.pieceToBeMoved.piece == "Pawn" and item.xLocation == self.whiteSideEnPasentPawnxLocation and isCapture and y.yLocation == 4):
                                    self.boardPieces[self.whiteSideEnPasentPawnxLocation - 1][3].piece = "Empty"
                                    self.boardPieces[self.whiteSideEnPasentPawnxLocation - 1][3].colour = "None"
                                    self.boardPieces[self.whiteSideEnPasentPawnxLocation - 1][3].value = 0
                                                    
                            else:
                                
                                # If the piece was set in a previous iteration as in en pasent position, it is no longer by the next move
                                if(self.whiteSideEnPasent):
                                    self.whiteSideEnPasent = False
                                
                                # Set if the pawn is in en pasent position
                                if(self.pieceToBeMoved.piece == "Pawn" and item.yLocation == 4 and y.yLocation == 2):
                                    self.whiteSideEnPasent = True
                                    self.whiteSideEnPasentPawnxLocation = y.xLocation  
                                    pass 
                                
                                # Remove the pawn being captured if done through en pasent
                                if(self.blackSideEnPasent and self.pieceToBeMoved.piece == "Pawn" and item.xLocation == self.blackSideEnPasentPawnxLocation and isCapture and y.yLocation == 5):
                                    self.boardPieces[self.blackSideEnPasentPawnxLocation - 1][4].piece = "Empty"
                                    self.boardPieces[self.blackSideEnPasentPawnxLocation - 1][4].colour = "None"
                                    self.boardPieces[self.blackSideEnPasentPawnxLocation - 1][4].value = 0
                            
                            # Increment the move number to simulate a move being made
                            self.numMove += 1
                            # Call the minimizer to make the next move
                            score = self.minimizer(curDepth + 1,depthLimit,oppositeColour,alpha, beta)
                            self.numMove -= 1
                            
                            
                            if (self.numMove == 37):  
                                with open('Unfiltered_Full.txt', 'a') as file:
                                    file.write("FINAL MOVE: {}, {}, {}, {}, {}\n".format(score, item.piece, item.colour, item.xLocation, item.yLocation))
                            '''
                            if (self.numMove == 11):
                                print ("FINAL MOVE: ", score, item.piece, item.colour, item.xLocation, item.yLocation)
                            '''
                              
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
            return self.evaluateBoard()
        
        isLegal = False
        isCapture = False
        lowestScore = 99999999
        castleFlag = False
        
        # Create a pre-move copy of the board
        boardCopy = copy.deepcopy(self.boardPieces)
        
        # Loop through all pieces on the board
        for x in self.boardPieces:
            for y in x:
                
                # Check if the chosen piece is of the active player
                if (y.colour == evalColour):
                    
                    # Acquire list of possible moves for the given piece
                    moves = []
                    self.moveAppender(moves,y,evalColour)
                    
                    # Loop through list of moves
                    for item in moves:
                        
                        # Set the piece to be moved as the current piece
                        self.pieceToBeMoved.piece = self.boardPieces[y.xLocation - 1][y.yLocation - 1].piece
                        self.pieceToBeMoved.colour = self.boardPieces[y.xLocation - 1][y.yLocation - 1].colour
                        self.pieceToBeMoved.value = self.boardPieces[y.xLocation - 1][y.yLocation - 1].value
                        self.pieceToBeMoved.xLocation = y.xLocation
                        self.pieceToBeMoved.yLocation = y.yLocation
                        '''
                        if (self.numMove == 4):
                            with open('Unfiltered_Full.txt', 'a') as file:
                                file.write("MAX CHOSEN: {}, {}, {}, {}\n".format(item.piece, item.colour, item.xLocation, item.yLocation))
                        '''
                        # Check if moving this piece to each move's location is legal
                        isLegal,isCapture = self.isLegalMove(self.boardPieces[item.xLocation - 1][item.yLocation - 1],item.xLocation,item.yLocation,evalColour)

                        if(isLegal):
                            
                            # If legal, make the move
                            self.boardPieces[item.xLocation - 1][item.yLocation - 1].piece = item.piece
                            self.boardPieces[item.xLocation - 1][item.yLocation - 1].colour = item.colour
                            self.boardPieces[item.xLocation - 1][item.yLocation - 1].value = item.value
                            
                            # Replace starting position with empty square
                            self.boardPieces[y.xLocation - 1][y.yLocation - 1].piece = "Empty"
                            self.boardPieces[y.xLocation - 1][y.yLocation - 1].colour = "None"
                            self.boardPieces[y.xLocation - 1][y.yLocation - 1].value = 0
                            
                            # Check if the move made is a castling move
                            if(self.isCastle):
                                
                                # Set the castle flag within the scope of the current move
                                castleFlag = True
                                if(evalColour == "Black"):
                                    
                                    self.blackHasCastled = True
                                    
                                    # Move the rook to complete kingside castling for black
                                    if(item.xLocation == 7):
                                        
                                        self.boardPieces[5][7].piece = self.boardPieces[7][7].piece
                                        self.boardPieces[5][7].colour = evalColour
                                        self.boardPieces[5][7].value = self.boardPieces[7][7].value
                                        
                                        self.boardPieces[7][7].piece = "Empty"
                                        self.boardPieces[7][7].colour = "None"
                                        self.boardPieces[7][7].value = 0
                                        
                                    # Move the rook to complete queenside castling for black
                                    elif(item.xLocation == 3):
                                        
                                        self.boardPieces[3][7].piece = self.boardPieces[0][7].piece
                                        self.boardPieces[3][7].colour = evalColour
                                        self.boardPieces[3][7].value = self.boardPieces[0][7].value
                                        
                                        self.boardPieces[0][7].piece = "Empty"
                                        self.boardPieces[0][7].colour = "None"
                                        self.boardPieces[0][7].value = 0
                                    
                                else:
                                    
                                    self.whiteHasCastled = True
                                    
                                    # Move the rook to complete kingside castling for white
                                    if(item.xLocation == 7):
                                        
                                        self.boardPieces[5][0].piece = self.boardPieces[7][0].piece
                                        self.boardPieces[5][0].colour = evalColour
                                        self.boardPieces[5][0].value = self.boardPieces[7][0].value
                                        
                                        self.boardPieces[7][0].piece = "Empty"
                                        self.boardPieces[7][0].colour = "None"
                                        self.boardPieces[7][0].value = 0
                                    
                                    # Move the rook to complete queenside castling for white
                                    elif(item.xLocation == 3):
                                        
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
                                if(self.pieceToBeMoved.piece == "Pawn" and item.yLocation == 5 and y.yLocation == 7):
                                    self.blackSideEnPasent = True
                                    self.blackSideEnPasentPawnxLocation = y.xLocation
                                    pass
                                
                                # Remove the pawn being captured if done through en pasent
                                if(self.whiteSideEnPasent and self.pieceToBeMoved.piece == "Pawn" and item.xLocation == self.whiteSideEnPasentPawnxLocation and isCapture and y.yLocation == 4):
                                    self.boardPieces[self.whiteSideEnPasentPawnxLocation - 1][3].piece = "Empty"
                                    self.boardPieces[self.whiteSideEnPasentPawnxLocation - 1][3].colour = "None"
                                    self.boardPieces[self.whiteSideEnPasentPawnxLocation - 1][3].value = 0
                                                    
                            else:
                                
                                # If the piece was set in a previous iteration as in en pasent position, it is no longer by the next move
                                if(self.whiteSideEnPasent):
                                    self.whiteSideEnPasent = False
                                
                                # Set if the pawn is in en pasent position
                                if(self.pieceToBeMoved.piece == "Pawn" and item.yLocation == 4 and y.yLocation == 2):
                                    self.whiteSideEnPasent = True
                                    self.whiteSideEnPasentPawnxLocation = y.xLocation  
                                    pass 
                                
                                # Remove the pawn being captured if done through en pasent
                                if(self.blackSideEnPasent and self.pieceToBeMoved.piece == "Pawn" and item.xLocation == self.blackSideEnPasentPawnxLocation and isCapture and y.yLocation == 5):
                                    self.boardPieces[self.blackSideEnPasentPawnxLocation - 1][4].piece = "Empty"
                                    self.boardPieces[self.blackSideEnPasentPawnxLocation - 1][4].colour = "None"
                                    self.boardPieces[self.blackSideEnPasentPawnxLocation - 1][4].value = 0

                            # Increment the move number to simulate a move being made
                            self.numMove += 1
                            # Call the maximizer to make the next move
                            score = self.maximizer(curDepth + 1,depthLimit,oppositeColour, alpha, beta)
                            self.numMove -= 1
                            
                            if (self.numMove == 36):
                                with open('Unfiltered_Full.txt', 'a') as file:
                                    file.write("MAX CHOSEN: {}, {}, {}, {}, {}\n".format(score, item.piece, item.colour, item.xLocation, item.yLocation))
                            
                            '''
                            if (self.numMove == 11):
                                print ("MAX CHOSEN: ", score, item.piece, item.colour, item.xLocation, item.yLocation)
                            '''
                            
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

    # Function to find out if a player is in checkmate
    def isCheckMate(self,colour):
        isLegal = False
        isCapture = False
        moves = []
        boardCopy = copy.deepcopy(self.boardPieces)

        # Makes all possible moves to see if it can avoid being in check
        for x in self.boardPieces:
            for y in x:
                if (y.colour == colour and not(y.piece == "Empty")):
                    
                    # Acquire list of possible moves for the given piece
                    moves = []
                    self.moveAppender(moves,y,colour)
                    
                    # Loop through list of moves
                    for item in moves:
                        
                        # Set the piece to be moved as the current piece
                        self.pieceToBeMoved.piece = self.boardPieces[y.xLocation - 1][y.yLocation - 1].piece
                        self.pieceToBeMoved.colour = self.boardPieces[y.xLocation - 1][y.yLocation - 1].colour
                        self.pieceToBeMoved.xLocation = self.boardPieces[y.xLocation - 1][y.yLocation - 1].xLocation
                        self.pieceToBeMoved.yLocation = self.boardPieces[y.xLocation - 1][y.yLocation - 1].yLocation
                        
                        # Check if moving this piece to each move's location is legal
                        isLegal,isCapture = self.isLegalMove(self.boardPieces[item.xLocation - 1][item.yLocation - 1],item.xLocation,item.yLocation,colour)

                        if(isLegal):
                            
                            # If legal, make the move
                            self.boardPieces[item.xLocation - 1][item.yLocation - 1] = item
                            self.boardPieces[y.xLocation - 1][y.yLocation - 1].piece = "Empty"
                            self.boardPieces[y.xLocation - 1][y.yLocation - 1].colour = "None"
                            
                            # Check if the player is still in check after the move
                            if(not(self.isInCheck(colour))):
                                
                                # If the player is not in check after a move, then the player is not in checkmate
                                self.boardPieces = copy.deepcopy(boardCopy)
                                return False
                            self.boardPieces = copy.deepcopy(boardCopy)
        return True
    
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
                        elif (self.whiteSideEnPasent and abs(y.xLocation - self.whiteSideEnPasentPawnxLocation) == 1):
                            moves.append(Pieces(y.piece,y.colour,y.xLocation + 1, y.yLocation - 1))
                        
                    # Black pawn capture to the left
                    if (y.xLocation > 1):  
                        if (self.boardPieces[y.xLocation - 2][y.yLocation - 2].colour == oppositeColour):                          
                            moves.append(Pieces(y.piece,y.colour,y.xLocation - 1, y.yLocation - 1)) 
                        elif (self.whiteSideEnPasent and abs(y.xLocation - self.whiteSideEnPasentPawnxLocation) == 1):
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
                    
board = Layout()
board.drawboard()
#board.test()
board.mainloop()   
