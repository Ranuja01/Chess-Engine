import tkinter as tk
import copy 
import easygui
import time

from PIL import ImageTk, Image

class Pieces():
    
    def __init__(self,pieceType,colour,x,y):
        self.piece = pieceType
        self.colour = colour
        self.xLocation = x
        self.yLocation = y
        self.value = 0
        
        if(self.piece == "Pawn"):
            self.value = 100 
        elif(self.piece == "Knight"):
            self.value = 270  
        elif(self.piece == "Bishop"):
            self.value = 300 
        elif(self.piece == "Rook"):
            self.value = 500   
        elif(self.piece == "Queen"):
            self.value = 900                      

class Layout(tk.Tk):
    colours = ["#ffffff", "#a72f03"]#square colours light then dark
            
    def __init__(self, n=8):
        super().__init__()
        self.n = n
        self.leftframe = tk.Frame(self)
        self.leftframe.grid(row=0, column=0, rowspan=10, padx=40)
        self.middleframe = tk.Frame(self)
        self.middleframe.grid(row=0, column=8, rowspan=8)
        self.canvas = tk.Canvas(self, width=950, height=768, )
        self.canvas.grid(row=0, column=1, columnspan=8, rowspan=8)
        self.board = [[None for row in range(n)] for col in range(n)]
        self.move = False
        self.pieceToBeMoved = None
        self.pawnToBePromoted = None
        self.numMove = 1
        self.whiteKingHasMoved = False
        self.blackKingHasMoved = False
        self.queenSideWhiteRookHasMoved = False
        self.kingSideWhiteRookHasMoved = False
        self.queenSideBlackRookHasMoved = False
        self.kingSideBlackRookHasMoved = False
        self.blackSideEnPasent = False
        self.blackSideEnPasentPawnxLocation = 0
        self.whiteSideEnPasent = False
        self.whiteSideEnPasentPawnxLocation = 0
        self.isPromotion = False
        self.isCastle = False
        self.isCheck = False
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
        
        
    def drawboard(self):
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
        
        
        # self.board[1][3] = self.canvas.create_rectangle(0, 450, 90, 540, fill="#%02x%02x%02x" % (0, 0, 0), tags=f"tile{1}{3}")    
        # self.canvas.tag_bind(f"tile{1}{3}","<Button-1>", lambda e, i=col+1, j=row+1: self.get_location(e,123,321))    
        
        blackPawnWhiteSquare = Image.open('Images/BlackPawnWhiteSquare.png')
        blackPawnWhiteSquare = blackPawnWhiteSquare.resize((90, 90), Image.ANTIALIAS)
        self.canvas.blackPawnWhiteSquareImage = ImageTk.PhotoImage(blackPawnWhiteSquare,master = self)
        
        blackPawnBlackSquare = Image.open('Images/BlackPawnBlackSquare.png')
        blackPawnBlackSquare = blackPawnBlackSquare.resize((90, 90), Image.ANTIALIAS)
        self.canvas.blackPawnBlackSquareImage = ImageTk.PhotoImage(blackPawnBlackSquare,master = self)
        
        blackRookWhiteSquare = Image.open('Images/BlackRookWhiteSquare.png')
        blackRookWhiteSquare = blackRookWhiteSquare.resize((90, 90), Image.ANTIALIAS)
        self.canvas.blackRookWhiteSquareImage = ImageTk.PhotoImage(blackRookWhiteSquare,master = self)
        
        blackKingWhiteSquare = Image.open('Images/BlackKingWhiteSquare.png')
        blackKingWhiteSquare = blackKingWhiteSquare.resize((90, 90), Image.ANTIALIAS)
        self.canvas.blackKingWhiteSquareImage = ImageTk.PhotoImage(blackKingWhiteSquare,master = self)
        
        blackKingBlackSquare = Image.open('Images/BlackKingBlackSquare.png')
        blackKingBlackSquare = blackKingBlackSquare.resize((90, 90), Image.ANTIALIAS)
        self.canvas.blackKingBlackSquareImage = ImageTk.PhotoImage(blackKingBlackSquare,master = self)
        
        blackKnightWhiteSquare = Image.open('Images/BlackKnightWhiteSquare.png')
        blackKnightWhiteSquare = blackKnightWhiteSquare.resize((90, 90), Image.ANTIALIAS)
        self.canvas.blackKnightWhiteSquareImage = ImageTk.PhotoImage(blackKnightWhiteSquare,master = self)
        
        blackKnightBlackSquare = Image.open('Images/BlackKnightBlackSquare.png')
        blackKnightBlackSquare = blackKnightBlackSquare.resize((90, 90), Image.ANTIALIAS)
        self.canvas.blackKnightBlackSquareImage = ImageTk.PhotoImage(blackKnightBlackSquare,master = self)
        
        blackRookBlackSquare = Image.open('Images/BlackRookBlackSquare.png')
        blackRookBlackSquare = blackRookBlackSquare.resize((90, 90), Image.ANTIALIAS)
        self.canvas.blackRookBlackSquareImage = ImageTk.PhotoImage(blackRookBlackSquare,master = self)
        
        blackQueenBlackSquare = Image.open('Images/BlackQueenBlackSquare.png')
        blackQueenBlackSquare = blackQueenBlackSquare.resize((90, 90), Image.ANTIALIAS)
        self.canvas.blackQueenBlackSquareImage = ImageTk.PhotoImage(blackQueenBlackSquare,master = self)
        
        blackQueenWhiteSquare = Image.open('Images/BlackQueenWhiteSquare.png')
        blackQueenWhiteSquare = blackQueenWhiteSquare.resize((90, 90), Image.ANTIALIAS)
        self.canvas.blackQueenWhiteSquareImage = ImageTk.PhotoImage(blackQueenWhiteSquare,master = self)
        
        blackBishopWhiteSquare = Image.open('Images/BlackBishopWhiteSquare.png')
        blackBishopWhiteSquare = blackBishopWhiteSquare.resize((90, 90), Image.ANTIALIAS)
        self.canvas.blackBishopWhiteSquareImage = ImageTk.PhotoImage(blackBishopWhiteSquare,master = self)
          
        blackBishopBlackSquare = Image.open('Images/BlackBishopBlackSquare.png')
        blackBishopBlackSquare = blackBishopBlackSquare.resize((90, 90), Image.ANTIALIAS)
        self.canvas.blackBishopBlackSquareImage = ImageTk.PhotoImage(blackBishopBlackSquare,master = self) 
        
        whitePawnWhiteSquare = Image.open('Images/whitePawnWhiteSquare.png')
        whitePawnWhiteSquare = whitePawnWhiteSquare.resize((90, 90), Image.ANTIALIAS)
        self.canvas.whitePawnWhiteSquareImage = ImageTk.PhotoImage(whitePawnWhiteSquare,master = self)

        whitePawnBlackSquare = Image.open('Images/whitePawnBlackSquare.png')
        whitePawnBlackSquare = whitePawnBlackSquare.resize((90, 90), Image.ANTIALIAS)
        self.canvas.whitePawnBlackSquareImage = ImageTk.PhotoImage(whitePawnBlackSquare,master = self)
        
        whiteRookWhiteSquare = Image.open('Images/whiteRookWhiteSquare.png')
        whiteRookWhiteSquare = whiteRookWhiteSquare.resize((90, 90), Image.ANTIALIAS)
        self.canvas.whiteRookWhiteSquareImage = ImageTk.PhotoImage(whiteRookWhiteSquare,master = self)
        
        whiteRookBlackSquare = Image.open('Images/whiteRookBlackSquare.png')
        whiteRookBlackSquare = whiteRookBlackSquare.resize((90, 90), Image.ANTIALIAS)
        self.canvas.whiteRookBlackSquareImage = ImageTk.PhotoImage(whiteRookBlackSquare,master = self)
        
        whiteKnightWhiteSquare = Image.open('Images/whiteKnightWhiteSquare.png')
        whiteKnightWhiteSquare = whiteKnightWhiteSquare.resize((90, 90), Image.ANTIALIAS)
        self.canvas.whiteKnightWhiteSquareImage = ImageTk.PhotoImage(whiteKnightWhiteSquare,master = self)
        
        whiteKnightBlackSquare = Image.open('Images/whiteKnightBlackSquare.png')
        whiteKnightBlackSquare = whiteKnightBlackSquare.resize((90, 90), Image.ANTIALIAS)
        self.canvas.whiteKnightBlackSquareImage = ImageTk.PhotoImage(whiteKnightBlackSquare,master = self)
        
        whiteKingBlackSquare = Image.open('Images/whiteKingBlackSquare.png')
        whiteKingBlackSquare = whiteKingBlackSquare.resize((90, 90), Image.ANTIALIAS)
        self.canvas.whiteKingBlackSquareImage = ImageTk.PhotoImage(whiteKingBlackSquare,master = self)
        
        whiteBishopWhiteSquare = Image.open('Images/whiteBishopWhiteSquare.png')
        whiteBishopWhiteSquare = whiteBishopWhiteSquare.resize((90, 90), Image.ANTIALIAS)
        self.canvas.whiteBishopWhiteSquareImage = ImageTk.PhotoImage(whiteBishopWhiteSquare,master = self)
        
        whiteBishopBlackSquare = Image.open('Images/whiteBishopBlackSquare.png')
        whiteBishopBlackSquare = whiteBishopBlackSquare.resize((90, 90), Image.ANTIALIAS)
        self.canvas.whiteBishopBlackSquareImage = ImageTk.PhotoImage(whiteBishopBlackSquare,master = self)
        
        whiteQueenWhiteSquare = Image.open('Images/whiteQueenWhiteSquare.png')
        whiteQueenWhiteSquare = whiteQueenWhiteSquare.resize((90, 90), Image.ANTIALIAS)
        self.canvas.whiteQueenWhiteSquareImage = ImageTk.PhotoImage(whiteQueenWhiteSquare,master = self)
        
        whiteQueenBlackSquare = Image.open('Images/whiteQueenBlackSquare.png')
        whiteQueenBlackSquare = whiteQueenBlackSquare.resize((90, 90), Image.ANTIALIAS)
        self.canvas.whiteQueenBlackSquareImage = ImageTk.PhotoImage(whiteQueenBlackSquare,master = self)
        
        whiteKingWhiteSquare = Image.open('Images/whiteKingWhiteSquare.png')
        whiteKingWhiteSquare = whiteKingWhiteSquare.resize((90, 90), Image.ANTIALIAS)
        self.canvas.whiteKingWhiteSquareImage = ImageTk.PhotoImage(whiteKingWhiteSquare,master = self)
        
        whiteKingInCheck = Image.open('Images/WhiteKingInCheck.png')
        whiteKingInCheck = whiteKingInCheck.resize((90, 90), Image.ANTIALIAS)
        self.canvas.whiteKingInCheckImage = ImageTk.PhotoImage(whiteKingInCheck,master = self)
        
        blackKingInCheck = Image.open('Images/BlackKingInCheck.png')
        blackKingInCheck = blackKingInCheck.resize((90, 90), Image.ANTIALIAS)
        self.canvas.blackKingInCheckImage = ImageTk.PhotoImage(blackKingInCheck,master = self)
        
        """
        for x in self.boardPieces:
            for y in x:
                print(y.piece + "X: " + str(y.xLocation) + ",Y: " + str(y.yLocation))
       """ 
       # print(self.canvas.image)
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
    
    def evaluateBoard(self,colour):
        sum = 0
        if (colour == "Black"):
            oppositeColour = "White"
        else:
            oppositeColour = "Black"
        if(self.isInCheck(colour)):
            if(self.isCheckMate(colour)):
                sum = -100000
        elif(self.isInCheck(oppositeColour)):
            if(self.isCheckMate(oppositeColour)):
                sum = 100000
        
        for item in self.boardPieces:
            for square in item:
                if(square.colour == colour and not(square.piece == "King")):
                    sum += square.value
                    if(square.xLocation == 4 and square.yLocation == 4 or square.xLocation == 4 and square.yLocation == 5 or square.xLocation == 5 and square.yLocation == 4 or square.xLocation == 5 and square.yLocation == 5):
                        sum += 25
        
        
        
        return sum
    
    def promotion(self,event,pieceChosen,colour):
        print(colour + " " + pieceChosen)
        piece = None
        if(self.isPromotion):
            if (self.numMove % 2 == 1):
                if (colour == "Black"):
                    if (pieceChosen == "Queen"):
                        self.pawnToBePromoted.piece = "Queen"
                        if (self.findSquareColour(self.pawnToBePromoted.xLocation,self.pawnToBePromoted.yLocation) == "White"):   
                            piece = self.canvas.create_image(45 + (self.pawnToBePromoted.xLocation - 1)*90,45,image = self.canvas.blackQueenWhiteSquareImage, anchor = 'center')
                        else:
                            piece = self.canvas.create_image(45 + (self.pawnToBePromoted.xLocation - 1)*90,45,image = self.canvas.blackQueenBlackSquareImage, anchor = 'center')
                    elif(pieceChosen == "Rook"):
                        self.pawnToBePromoted.piece = "Rook"
                        if (self.findSquareColour(self.pawnToBePromoted.xLocation,self.pawnToBePromoted.yLocation) == "White"):   
                            piece = self.canvas.create_image(45 + (self.pawnToBePromoted.xLocation - 1)*90,45,image = self.canvas.blackRookWhiteSquareImage, anchor = 'center')
                        else:
                            piece = self.canvas.create_image(45 + (self.pawnToBePromoted.xLocation - 1)*90,45,image = self.canvas.blackRookBlackSquareImage, anchor = 'center')
                    elif(pieceChosen == "Bishop"):
                        self.pawnToBePromoted.piece = "Bishop"
                        if (self.findSquareColour(self.pawnToBePromoted.xLocation,self.pawnToBePromoted.yLocation) == "White"):   
                            piece = self.canvas.create_image(45 + (self.pawnToBePromoted.xLocation - 1)*90,45,image = self.canvas.blackBishopWhiteSquareImage, anchor = 'center')
                        else:
                            piece = self.canvas.create_image(45 + (self.pawnToBePromoted.xLocation - 1)*90,45,image = self.canvas.blackBishopBlackSquareImage, anchor = 'center')        
                    elif(pieceChosen == "Knight"):
                        self.pawnToBePromoted.piece = "Knight"
                        if (self.findSquareColour(self.pawnToBePromoted.xLocation,self.pawnToBePromoted.yLocation) == "White"):   
                            piece = self.canvas.create_image(45 + (self.pawnToBePromoted.xLocation - 1)*90,45,image = self.canvas.blackKnightWhiteSquareImage, anchor = 'center')
                        else:
                            piece = self.canvas.create_image(45 + (self.pawnToBePromoted.xLocation - 1)*90,45,image = self.canvas.blackKnightBlackSquareImage, anchor = 'center')       
            else:
                if (colour == "White"):
                    if (pieceChosen == "Queen"):
                        self.pawnToBePromoted.piece = "Queen"
                        if (self.findSquareColour(self.pawnToBePromoted.xLocation,self.pawnToBePromoted.yLocation) == "White"):   
                            piece = self.canvas.create_image(45 + (self.pawnToBePromoted.xLocation - 1)*90,45,image = self.canvas.whiteQueenWhiteSquareImage, anchor = 'center')
                        else:
                            piece = self.canvas.create_image(45 + (self.pawnToBePromoted.xLocation - 1)*90,45,image = self.canvas.whiteQueenBlackSquareImage, anchor = 'center')
                    elif(pieceChosen == "Rook"):
                        self.pawnToBePromoted.piece = "Rook"
                        if (self.findSquareColour(self.pawnToBePromoted.xLocation,self.pawnToBePromoted.yLocation) == "White"):   
                            piece = self.canvas.create_image(45 + (self.pawnToBePromoted.xLocation - 1)*90,45,image = self.canvas.whiteRookWhiteSquareImage, anchor = 'center')
                        else:
                            piece = self.canvas.create_image(45 + (self.pawnToBePromoted.xLocation - 1)*90,45,image = self.canvas.whiteRookBlackSquareImage, anchor = 'center')
                    elif(pieceChosen == "Bishop"):
                        self.pawnToBePromoted.piece = "Bishop"
                        if (self.findSquareColour(self.pawnToBePromoted.xLocation,self.pawnToBePromoted.yLocation) == "White"):   
                            piece = self.canvas.create_image(45 + (self.pawnToBePromoted.xLocation - 1)*90,45,image = self.canvas.whiteBishopWhiteSquareImage, anchor = 'center')
                        else:
                            piece = self.canvas.create_image(45 + (self.pawnToBePromoted.xLocation - 1)*90,45,image = self.canvas.whiteBishopBlackSquareImage, anchor = 'center')        
                    elif(pieceChosen == "Knight"):
                        self.pawnToBePromoted.piece = "Knight"
                        if (self.findSquareColour(self.pawnToBePromoted.xLocation,self.pawnToBePromoted.yLocation) == "White"):   
                            piece = self.canvas.create_image(45 + (self.pawnToBePromoted.xLocation - 1)*90,45,image = self.canvas.whiteKnightWhiteSquareImage, anchor = 'center')
                        else:
                            piece = self.canvas.create_image(45 + (self.pawnToBePromoted.xLocation - 1)*90,45,image = self.canvas.whiteKnightBlackSquareImage, anchor = 'center') 
            try:
                self.canvas.tag_bind(piece,"<Button-1>", lambda e, i=self.pawnToBePromoted.xLocation, j=self.pawnToBePromoted.yLocation: self.get_location(e,i,j))
                self.boardPieces[self.pawnToBePromoted.xLocation - 1][self.pawnToBePromoted.yLocation - 1] == self.pawnToBePromoted
                self.isPromotion = False
                if (self.numMove % 2 == 1):
                    if(self.isInCheck("White")):
                        #print("JOJOJ")
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
                if(self.isCheck): 
                    
                    pieces = self.canvas.create_image(45 + (kingPiece.xLocation-1)*90,45 + (8-kingPiece.yLocation)*90,image = newPiece, anchor = 'center')
                    self.canvas.tag_bind(pieces,"<Button-1>", lambda e, x = kingPiece.xLocation, y=kingPiece.yLocation: self.get_location(e,x,y))
                
            except:
                pass
        pass
    def isLegalPawnMove(self,curItem,i,j,colour):
        isLegal = False
        isCapture = False
        piece = None
        #print(self.pieceToBeMoved.xLocation,self.pieceToBeMoved.yLocation)
        if(curItem.piece == "Empty"):
            if (colour == "White"):
                if(self.pieceToBeMoved.yLocation == 2):
                    if(self.pieceToBeMoved.xLocation == i and (j == 3 or j == 4)):
                        if(j == 3 and self.boardPieces[i-1][2].piece == "Empty"):
                            isLegal = True
                        elif(j == 4 and self.boardPieces[i-1][2].piece == "Empty" and self.boardPieces[i-1][3].piece == "Empty"):
                            isLegal = True
                elif(self.pieceToBeMoved.xLocation == i and j - 1 == self.pieceToBeMoved.yLocation):
                    isLegal = True
                elif(self.blackSideEnPasent and abs(self.pieceToBeMoved.xLocation - self.blackSideEnPasentPawnxLocation) == 1 and (i + 1 == self.pieceToBeMoved.xLocation and j - 1 == self.pieceToBeMoved.yLocation or i - 1 == self.pieceToBeMoved.xLocation and j - 1 == self.pieceToBeMoved.yLocation)):
                    isLegal = True
                    isCapture = True
                    
                    
                if (isLegal):    
                    if (self.findSquareColour(i,j) =="White"):   
                        piece = self.canvas.whitePawnWhiteSquareImage
                    else:
                        piece = self.canvas.whitePawnBlackSquareImage
            else:
                
                if(self.pieceToBeMoved.yLocation == 7):
                    if(self.pieceToBeMoved.xLocation == i and (j == 6 or j == 5)):
                         if(j == 6 and self.boardPieces[i-1][5].piece == "Empty"):
                            isLegal = True
                         elif(j == 5 and self.boardPieces[i-1][5].piece == "Empty" and self.boardPieces[i-1][4].piece == "Empty"):
                            isLegal = True
                elif(self.pieceToBeMoved.xLocation == i and j + 1 == self.pieceToBeMoved.yLocation):
                    isLegal = True
                elif(self.whiteSideEnPasent and abs(self.pieceToBeMoved.xLocation - self.whiteSideEnPasentPawnxLocation) == 1 and (i + 1 == self.pieceToBeMoved.xLocation and j + 1 == self.pieceToBeMoved.yLocation or i - 1 == self.pieceToBeMoved.xLocation and j + 1 == self.pieceToBeMoved.yLocation)):
                    isLegal = True
                    isCapture = True 
                    
                if (isLegal):    
                    if (self.findSquareColour(i,j) =="White"):   
                        piece = self.canvas.blackPawnWhiteSquareImage
                    else:
                        piece = self.canvas.blackPawnBlackSquareImage

            
        else:

            if (colour == "White"):
                
                if(curItem.colour == "Black" and (i + 1 == self.pieceToBeMoved.xLocation and j - 1 == self.pieceToBeMoved.yLocation or i - 1 == self.pieceToBeMoved.xLocation and j - 1 == self.pieceToBeMoved.yLocation)):
                    if (self.findSquareColour(i,j) =="White"):   
                        piece = self.canvas.whitePawnWhiteSquareImage
                    else:
                        piece = self.canvas.whitePawnBlackSquareImage
                    isLegal = True
                    isCapture = True
            else:

                if(curItem.colour == "White" and (i + 1 == self.pieceToBeMoved.xLocation and j + 1 == self.pieceToBeMoved.yLocation or i - 1 == self.pieceToBeMoved.xLocation and j + 1 == self.pieceToBeMoved.yLocation)):
                    if (self.findSquareColour(i,j) =="White"):   
                        piece = self.canvas.blackPawnWhiteSquareImage
                    else:
                        piece = self.canvas.blackPawnBlackSquareImage
                    isLegal = True
                    isCapture = True
        
        if(self.move):
            return piece,isLegal,isCapture
        else:
            return isLegal,isCapture
        
    def isLegalKnightMove(self,curItem,i,j,colour):
        isLegal = False
        isCapture = False
        piece = None
        #print(self.pieceToBeMoved.xLocation,self.pieceToBeMoved.yLocation)
        if (abs(i - self.pieceToBeMoved.xLocation) == 2 and abs(j - self.pieceToBeMoved.yLocation) == 1 or abs(i - self.pieceToBeMoved.xLocation) == 1 and abs(j - self.pieceToBeMoved.yLocation) == 2):
            if (colour == "White"):
                
                if (self.findSquareColour(i,j) =="White"):   
                    piece = self.canvas.whiteKnightWhiteSquareImage
                else:
                    piece = self.canvas.whiteKnightBlackSquareImage
                isLegal = True
                if(curItem.colour == "Black"):
                    isCapture = True
            else:
                
                if (self.findSquareColour(i,j) =="White"):   
                    piece = self.canvas.blackKnightWhiteSquareImage
                else:
                    piece = self.canvas.blackKnightBlackSquareImage
                isLegal = True
                if(curItem.colour == "White"):
                    isCapture = True
        if(self.move):
            return piece,isLegal,isCapture
        else:
            return isLegal,isCapture
    
    def isLegalBishopMove(self,curItem,i,j,colour):
        isLegal = False
        isCapture = False
        piece = None
        #print(self.pieceToBeMoved.xLocation,self.pieceToBeMoved.yLocation)
        if (abs(i - self.pieceToBeMoved.xLocation) ==abs(j - self.pieceToBeMoved.yLocation)):
            
            xMultiplier = 1
            yMultiplier = 1
            isLegal = True
            
            if(i > self.pieceToBeMoved.xLocation):
                xMultiplier = -1
            if (j > self.pieceToBeMoved.yLocation):
                yMultiplier = -1
            for square in range(1,abs(i - self.pieceToBeMoved.xLocation)):
                
                if(not(self.boardPieces[i + square*xMultiplier - 1][j + square*yMultiplier - 1].piece == "Empty")):
                    isLegal = False
                    
            if(isLegal):
                isLegal = False
                if (colour == "White"):
                    
                    if (self.findSquareColour(i,j) =="White"):   
                        piece = self.canvas.whiteBishopWhiteSquareImage
                    else:
                        piece = self.canvas.whiteBishopBlackSquareImage
                    isLegal = True
                    if(curItem.colour == "Black"):
                        isCapture = True   
                else:
                    
                    if (self.findSquareColour(i,j) =="White"):   
                        piece = self.canvas.blackBishopWhiteSquareImage
                    else:
                        piece = self.canvas.blackBishopBlackSquareImage
                    isLegal = True
                    if(curItem.colour == "White"):
                        isCapture = True  
        if(self.move):
            return piece,isLegal,isCapture
        else:
            return isLegal,isCapture
    
    def isLegalRookMove(self,curItem,i,j,colour):
        isLegal = False
        isCapture = False
        piece = None
        #print(self.pieceToBeMoved.xLocation,self.pieceToBeMoved.yLocation)
        if(i == self.pieceToBeMoved.xLocation or j == self.pieceToBeMoved.yLocation):
            
            xMultiplier = 1
            yMultiplier = 1
            isLegal = True
            
            if (i == self.pieceToBeMoved.xLocation):
                if (j > self.pieceToBeMoved.yLocation):
                    yMultiplier = -1
                for square in range(1,abs(j - self.pieceToBeMoved.yLocation)):
                
                    if(not(self.boardPieces[i-1][j + square*yMultiplier - 1].piece == "Empty")):
                        isLegal = False
                
            else:
                if(i > self.pieceToBeMoved.xLocation):
                    xMultiplier = -1
            
                for square in range(1,abs(i - self.pieceToBeMoved.xLocation)):
                    #print (i+ square*xMultiplier - 1)
                    if(not(self.boardPieces[i+ square*xMultiplier - 1][j-1].piece == "Empty")):
                        isLegal = False
            
            if (isLegal):
                isLegal = False
                #print("WAAAAAAAAAAA")
                #print(self.numMove)
                if (colour == "White"):
                    
                    if (self.findSquareColour(i,j) =="White"):   
                        piece = self.canvas.whiteRookWhiteSquareImage
                    else:
                        piece = self.canvas.whiteRookBlackSquareImage
                    isLegal = True
                    if(curItem.colour == "Black"):
                        isCapture = True   
                else:
                    
                    if (self.findSquareColour(i,j) =="White"):   
                        piece = self.canvas.blackRookWhiteSquareImage
                    else:
                        piece = self.canvas.blackRookBlackSquareImage
                    isLegal = True
                    if(curItem.colour == "White"):
                        isCapture = True 
                        
                      
                        
        if(self.move):
            return piece,isLegal,isCapture
        else:
            return isLegal,isCapture
    
    def isLegalQueenMove(self,curItem,i,j,colour):
        isLegal = False
        isCapture = False
        piece = None
        #print(self.pieceToBeMoved.xLocation,self.pieceToBeMoved.yLocation)
        xMultiplier = 1
        yMultiplier = 1
        isLegal = True
            
            
        if (abs(i - self.pieceToBeMoved.xLocation) ==abs(j - self.pieceToBeMoved.yLocation)):
            if(i > self.pieceToBeMoved.xLocation):
                xMultiplier = -1
            if (j > self.pieceToBeMoved.yLocation):
                yMultiplier = -1
            for square in range(1,abs(i - self.pieceToBeMoved.xLocation)):
                
                if(not(self.boardPieces[i + square*xMultiplier - 1][j + square*yMultiplier - 1].piece == "Empty")):
                    isLegal = False
        elif(i == self.pieceToBeMoved.xLocation or j == self.pieceToBeMoved.yLocation):
            if (i == self.pieceToBeMoved.xLocation):
                if (j > self.pieceToBeMoved.yLocation):
                    yMultiplier = -1
                for square in range(1,abs(j - self.pieceToBeMoved.yLocation)):
                
                    if(not(self.boardPieces[i-1][j + square*yMultiplier - 1].piece == "Empty")):
                        isLegal = False
            else:
                if(i > self.pieceToBeMoved.xLocation):
                    xMultiplier = -1
            
                for square in range(1,abs(i - self.pieceToBeMoved.xLocation)):
                
                    if(not(self.boardPieces[i+ square*xMultiplier - 1][j-1].piece == "Empty")):
                        isLegal = False
        else:
            isLegal = False
        
        if (isLegal):
                isLegal = False
                if (colour == "White"):
                    
                    if (self.findSquareColour(i,j) =="White"):   
                        piece = self.canvas.whiteQueenWhiteSquareImage
                    else:
                        piece = self.canvas.whiteQueenBlackSquareImage
                    isLegal = True
                    if(curItem.colour == "Black"):
                        isCapture = True   
                else:
                    
                    if (self.findSquareColour(i,j) =="White"):   
                        piece = self.canvas.blackQueenWhiteSquareImage
                    else:
                        piece = self.canvas.blackQueenBlackSquareImage
                    isLegal = True
                    if(curItem.colour == "White"):
                        isCapture = True 
        if(self.move):
            return piece,isLegal,isCapture
        else:
            return isLegal,isCapture
    
    def isLegalKingMove(self,curItem,i,j,colour):
        isLegal = False
        isCapture = False
        piece = None
        #print(self.pieceToBeMoved.xLocation,self.pieceToBeMoved.yLocation)
        if((abs(i - self.pieceToBeMoved.xLocation) == 1 and j == self.pieceToBeMoved.yLocation) or (abs(j - self.pieceToBeMoved.yLocation) == 1 and i == self.pieceToBeMoved.xLocation) or (abs(i - self.pieceToBeMoved.xLocation) == 1 and abs(j - self.pieceToBeMoved.yLocation) == 1)):
            if (colour == "White"):
                
                if (self.findSquareColour(i,j) =="White"):   
                    piece = self.canvas.whiteKingWhiteSquareImage
                else:
                    piece = self.canvas.whiteKingBlackSquareImage
                isLegal = True
                if(curItem.colour == "Black"):
                    isCapture = True   
            else:
                
                if (self.findSquareColour(i,j) =="White"):   
                    piece = self.canvas.blackKingWhiteSquareImage
                else:
                    piece = self.canvas.blackKingBlackSquareImage
                isLegal = True
                if(curItem.colour == "White"):
                    isCapture = True
        else:
            if (colour == "White"):
                if (i == 7 and j == 1):
                    
                    if(not(self.whiteKingHasMoved) and not(self.kingSideWhiteRookHasMoved)):
                        
                        if (self.boardPieces[5][0].piece == "Empty" and self.boardPieces[6][0].piece == "Empty" and self.boardPieces[7][0].piece == "Rook" and self.boardPieces[7][0].colour == "White"):
                            self.isCastle = True
                            
                            for x in self.boardPieces:
                                for y in x:
                                    if (y.colour == "Black"):
                                        if(self.IsAttacking(y,self.boardPieces[4][0]) or self.IsAttacking(y,self.boardPieces[5][0]) or self.IsAttacking(y,self.boardPieces[6][0])):
                                             self.isCastle = False
                                             
                if (i == 3 and j == 1):
                    if(not(self.whiteKingHasMoved) and not(self.queenSideWhiteRookHasMoved)):
                        if (self.boardPieces[1][0].piece == "Empty" and self.boardPieces[2][0].piece == "Empty" and self.boardPieces[3][0].piece == "Empty" and self.boardPieces[0][0].piece == "Rook" and self.boardPieces[0][0].colour == "White"):
                            self.isCastle = True
                            for x in self.boardPieces:
                                for y in x:
                                    if (y.colour == "Black"):
                                        if(self.IsAttacking(y,self.boardPieces[4][0]) or self.IsAttacking(y,self.boardPieces[1][0]) or self.IsAttacking(y,self.boardPieces[2][0]) or self.IsAttacking(y,self.boardPieces[3][0])):
                                             self.isCastle = False
            else:
                if (i == 7 and j == 8):
                    print("A")
                    if(not(self.blackKingHasMoved) and not(self.kingSideBlackRookHasMoved)):
                        print("B")
                        if (self.boardPieces[5][7].piece == "Empty" and self.boardPieces[6][7].piece == "Empty" and self.boardPieces[7][7].piece == "Rook" and self.boardPieces[7][7].colour == "Black"):
                            self.isCastle = True
                            print("C")
                            for x in self.boardPieces:
                                for y in x:
                                    if (y.colour == "White"):
                                        if(self.IsAttacking(y,self.boardPieces[4][7]) or self.IsAttacking(y,self.boardPieces[5][7]) or self.IsAttacking(y,self.boardPieces[6][7])):
                                             self.isCastle = False
                                             print("D")
                if (i == 3 and j == 8):
                    if(not(self.blackKingHasMoved) and not(self.queenSideBlackRookHasMoved)):
                        if (self.boardPieces[1][7].piece == "Empty" and self.boardPieces[2][7].piece == "Empty" and self.boardPieces[3][7].piece == "Empty" and self.boardPieces[0][7].piece == "Rook" and self.boardPieces[0][7].colour == "Black"):
                            self.isCastle = True
                            for x in self.boardPieces:
                                for y in x:
                                    if (y.colour == "White"):
                                        if(self.IsAttacking(y,self.boardPieces[4][7]) or self.IsAttacking(y,self.boardPieces[1][7]) or self.IsAttacking(y,self.boardPieces[2][7]) or self.IsAttacking(y,self.boardPieces[3][7])):
                                             self.isCastle = False
            if(self.isCastle):
                #print("DFGHJKL;LKJHGFD")
                isLegal = True
                if(i == 7 and j == 1):
                    piece = self.canvas.whiteKingBlackSquareImage
                    self.whiteKingHasMoved = True
                    self.kingSideWhiteRookHasMoved = True
                elif (i == 3 and j == 1):
                    piece = self.canvas.whiteKingBlackSquareImage
                    self.whiteKingHasMoved = True
                    self.queenSideWhiteRookHasMoved = True
                elif (i == 7 and j == 8):
                    piece = self.canvas.blackKingWhiteSquareImage
                    self.blackKingHasMoved = True
                    self.kingSideBlackRookHasMoved = True
                elif (i == 3 and j == 8):
                    piece = self.canvas.blackKingWhiteSquareImage
                    self.blackKingHasMoved = True
                    self.queenSideBlackRookHasMoved = True 
        if(self.move):
            return piece,isLegal,isCapture
        else:
            return isLegal,isCapture
    
    def isLegalMove(self,curItem,i,j,colour):
        if(self.pieceToBeMoved.colour == colour and not(curItem.colour == colour)):
            if(self.pieceToBeMoved.piece == "Pawn"):
                #print("AAAA")
                return self.isLegalPawnMove(curItem,i,j,colour)     
            elif(self.pieceToBeMoved.piece == "Knight"):
                return self.isLegalKnightMove(curItem,i,j,colour) 
            elif(self.pieceToBeMoved.piece == "Bishop"):
                return self.isLegalBishopMove(curItem,i,j,colour) 
            elif(self.pieceToBeMoved.piece == "Rook"):
                return self.isLegalRookMove(curItem,i,j,colour)  
            elif(self.pieceToBeMoved.piece == "Queen"):
                return self.isLegalQueenMove(curItem,i,j,colour)                     
            elif(self.pieceToBeMoved.piece == "King"):
                return self.isLegalKingMove(curItem,i,j,colour)
        else:
            if(self.move):
                return None,False,False
            else:
                return False,False
    
    def get_location(self, event, i, j):
        print (i, j)
        
        isLegal = False
        isCapture = False
        piece = None
        
        if (not(self.isPromotion)):
            
            if (self.move):
                
    
                #easygui.msgbox("HEEELOOOOOOO", title="simple gui")
                curItem = self.boardPieces[i-1][j-1] 
                curColour = self.pieceToBeMoved.colour
                piece,isLegal,isCapture = self.isLegalMove(curItem,i,j,curColour)
                
                if (self.numMove % 2 == 1):
                    if(self.blackSideEnPasent and self.pieceToBeMoved.piece == "Pawn" and i == self.blackSideEnPasentPawnxLocation and isCapture and self.pieceToBeMoved.yLocation == 5):
                        print("DDDDWE")
                        curItem = self.boardPieces[self.blackSideEnPasentPawnxLocation - 1][4]
                        
                else:
                    if(self.whiteSideEnPasent and self.pieceToBeMoved.piece == "Pawn" and i == self.whiteSideEnPasentPawnxLocation and isCapture and self.pieceToBeMoved.yLocation == 4):
                        curItem = self.boardPieces[self.whiteSideEnPasentPawnxLocation - 1][3]
                        
              
                if(isLegal):

                    curItem.xLocation = self.pieceToBeMoved.xLocation
                    curItem.yLocation = self.pieceToBeMoved.yLocation
                    
                    temp = copy.copy(curItem)
                    
                    self.pieceToBeMoved.xLocation = i
                    self.pieceToBeMoved.yLocation = j
                    
                    self.boardPieces[i-1][j-1] = self.pieceToBeMoved
                    self.boardPieces[temp.xLocation - 1][temp.yLocation-1] = temp
                    
                    if(isCapture):
                        #print("WAAAAA")
                        self.boardPieces[temp.xLocation - 1][temp.yLocation-1].piece = "Empty"
                        self.boardPieces[temp.xLocation - 1][temp.yLocation-1].colour = "None"
                        #self.boardPieces[temp.xLocation - 1][temp.yLocation-1].activity = False
                    
                    if (self.numMove % 2 == 1):
                        #print("MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM") 
                        if(self.isInCheck("White")):
                             
                             isLegal = False
                    else:
                        if(self.isInCheck("Black")):    
                             #print("TTTTTTTTT") 
                             isLegal = False
                    
                    if(isCapture):
                        self.boardPieces[temp.xLocation - 1][temp.yLocation-1].piece = curItem.piece
                        self.boardPieces[temp.xLocation - 1][temp.yLocation-1].colour = curItem.colour
                        #self.boardPieces[temp.xLocation - 1][temp.yLocation-1].active = curItem.active
                    
                    self.pieceToBeMoved.xLocation = temp.xLocation
                    self.pieceToBeMoved.yLocation = temp.yLocation
                    
                    temp.xLocation = i
                    temp.yLocation = j
                    
                    curitem = copy.copy(temp)
                    #print (curitem.piece + " " +  str(curItem.xLocation) + " " +  str(curItem.yLocation))
                    self.boardPieces[i-1][j-1] = curitem
                    #print (self.boardPieces[i-1][j-1].piece + " " +  str(self.boardPieces[i-1][j-1].xLocation) + " " +  str(self.boardPieces[i-1][j-1].yLocation))
                    self.boardPieces[self.pieceToBeMoved.xLocation - 1][self.pieceToBeMoved.yLocation-1] = self.pieceToBeMoved
                    print(isLegal)
                    if(isLegal): 
                        #print("AAAA")
                        
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
                        
                        
                        
                        x1 = (self.pieceToBeMoved.xLocation - 1) * 90
                        y1 = (8-self.pieceToBeMoved.yLocation) * 90
                        x2 = x1 + 90
                        y2 = y1 + 90
        
                        if self.findSquareColour(self.pieceToBeMoved.xLocation,self.pieceToBeMoved.yLocation) == "White":
                            colour = "#%02x%02x%02x" % (255,255,255)
                        else:
                            colour = "#%02x%02x%02x" % (167,47,3)
        
                        pieces = self.canvas.create_image(45 + (i-1)*90,45 + (8-j)*90,image = piece, anchor = 'center')
                        
                        self.canvas.tag_bind(pieces,"<Button-1>", lambda e, x = i, y=j: self.get_location(e,x,y))
            
                        self.board[i-1][j-1] = self.canvas.create_rectangle(x1, y1, x2, y2, fill=colour, tags=f"tile{self.pieceToBeMoved.xLocation}{self.pieceToBeMoved.yLocation}")    
                        self.canvas.tag_bind(f"tile{self.pieceToBeMoved.xLocation}{self.pieceToBeMoved.yLocation}","<Button-1>", lambda e, x=self.pieceToBeMoved.xLocation, y=self.pieceToBeMoved.yLocation: self.get_location(e,x,y))
                        
                        curItem.xLocation = self.pieceToBeMoved.xLocation
                        curItem.yLocation = self.pieceToBeMoved.yLocation
                        
                        temp = copy.copy(curItem)
                        
                        self.pieceToBeMoved.xLocation = i
                        self.pieceToBeMoved.yLocation = j
                        
                        self.boardPieces[i-1][j-1] = self.pieceToBeMoved
                        self.boardPieces[temp.xLocation - 1][temp.yLocation-1] = temp
                        
                        if (self.numMove % 2 == 1):
                            if(self.blackSideEnPasent and self.pieceToBeMoved.piece == "Pawn" and i == self.blackSideEnPasentPawnxLocation and isCapture and temp.yLocation == 5):
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
                                self.boardPieces[self.blackSideEnPasentPawnxLocation - 1][4].piece = "Empty"
                                self.boardPieces[self.blackSideEnPasentPawnxLocation - 1][4].colour = "None"
                        else:
                            if(self.whiteSideEnPasent and self.pieceToBeMoved.piece == "Pawn" and i == self.whiteSideEnPasentPawnxLocation and isCapture and temp.yLocation == 4):
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
                                self.boardPieces[self.whiteSideEnPasentPawnxLocation - 1][3].piece = "Empty"
                                self.boardPieces[self.whiteSideEnPasentPawnxLocation - 1][3].colour = "None"
                        
                        
                        
                        #isLegal = False
    
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
                            
                            pieces = self.canvas.create_image(45 + (rookToBeMoved.xLocation-1)*90,45 + (8-rookToBeMoved.yLocation)*90,image = piece, anchor = 'center')
                            self.canvas.tag_bind(pieces,"<Button-1>", lambda e, x = rookToBeMoved.xLocation, y=rookToBeMoved.yLocation: self.get_location(e,x,y))
                                                 
                            x1 = (emptySpace.xLocation - 1) * 90
                            y1 = (8-emptySpace.yLocation) * 90
                            x2 = x1 + 90
                            y2 = y1 + 90
                            
                            self.board[emptySpace.xLocation-1][emptySpace.yLocation-1] = self.canvas.create_rectangle(x1, y1, x2, y2, fill=colour, tags=f"tile{emptySpace.xLocation}{emptySpace.yLocation}")    
                            self.canvas.tag_bind(f"tile{emptySpace.xLocation}{emptySpace.yLocation}","<Button-1>", lambda e, x=emptySpace.xLocation, y=emptySpace.yLocation: self.get_location(e,x,y))
    
                            
                            self.boardPieces[emptySpace.xLocation-1][emptySpace.yLocation-1] = emptySpace
                            self.boardPieces[rookToBeMoved.xLocation - 1][rookToBeMoved.yLocation-1] = rookToBeMoved
                            self.isCastle = False
    
                        if(isCapture):
                            self.boardPieces[temp.xLocation - 1][temp.yLocation-1].piece = "Empty"
                            self.boardPieces[temp.xLocation - 1][temp.yLocation-1].colour = "None"
                            #self.boardPieces[temp.xLocation - 1][temp.yLocation-1].activity = False
                        
                        if (self.isCheck):
                            if (self.numMove % 2 == 1):
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
                        if (self.numMove % 2 == 1):
                            
                            if(self.whiteSideEnPasent):
                                self.whiteSideEnPasent = False
                            
                            if(self.pieceToBeMoved.piece == "Pawn" and self.pieceToBeMoved.yLocation == 4 and curItem.yLocation == 2):
                                self.whiteSideEnPasent = True
                                self.whiteSideEnPasentPawnxLocation = self.pieceToBeMoved.xLocation
                                print("QWEQWE")
                            
                            if(self.isInCheck("Black")):
                                #print("JOJOJ")
                                
                                if(self.isCheckMate("Black")):
                                    easygui.msgbox("White Wins!", title="Winner!")
                                
                                kingPiece = None
                                for x in self.boardPieces:
                                    for y in x:
                                        if (y.piece == "King" and y.colour == "Black"):
                                            kingPiece = y
                                            piece = self.canvas.blackKingInCheckImage
                                            self.isCheck = True
                            elif(self.pieceToBeMoved.piece == "Pawn" and j == 8):
                                self.isPromotion = True
                                self.pawnToBePromoted = self.pieceToBeMoved
                            
                            elif(self.isCheckMate("Black")):
                                easygui.msgbox("Draw by stalemate", title="Draw")
                              
                        else:
                            
                            
                            if(self.blackSideEnPasent):
                                self.blackSideEnPasent = False
                            
                            if(self.pieceToBeMoved.piece == "Pawn" and self.pieceToBeMoved.yLocation == 5 and curItem.yLocation == 7):
                                
                                self.blackSideEnPasent = True
                                self.blackSideEnPasentPawnxLocation = self.pieceToBeMoved.xLocation
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
                            elif(self.pieceToBeMoved.piece == "Pawn" and j == 1):
                                self.isPromotion = True
                                self.pawnToBePromoted = self.pieceToBeMoved
                                
                            elif(self.isCheckMate("White")):
                                easygui.msgbox("Draw by stalemate", title="Draw")
                                    
                        if(self.isCheck): 
                            
                            pieces = self.canvas.create_image(45 + (kingPiece.xLocation-1)*90,45 + (8-kingPiece.yLocation)*90,image = piece, anchor = 'center')
                            self.canvas.tag_bind(pieces,"<Button-1>", lambda e, x = kingPiece.xLocation, y=kingPiece.yLocation: self.get_location(e,x,y))
                        self.numMove += 1
                        board.update()
                
                
                self.move = False
            else:
                if (self.boardPieces[i-1][j-1].piece == "Empty"):
                    print(self.boardPieces[i-1][j-1].piece)
                else:            
                    print(self.boardPieces[i-1][j-1].colour + " " + self.boardPieces[i-1][j-1].piece)
                    if(self.numMove % 2 == 1 and self.boardPieces[i-1][j-1].colour == "White" or self.numMove % 2 == 0 and self.boardPieces[i-1][j-1].colour == "Black"):
                        self.pieceToBeMoved = self.boardPieces[i-1][j-1]
                        self.move = True
    
    
        
    def findSquareColour(self,i,j):
        if (i + j) % 2 == 0:
            return "Black"
        else:
            return "White"
        
    def isInCheck(self,colour):
        kingPiece = None
        for x in self.boardPieces:
            for y in x:
                if (y.piece == "King" and y.colour == colour):
                    kingPiece = y
        """
        print(self.boardPieces[4][7].piece + self.boardPieces[4][7].colour)
        print(self.boardPieces[5][7].piece + self.boardPieces[5][7].colour)
        print(kingPiece.piece + " DDDDD " + kingPiece.colour)
        """
        for x in self.boardPieces:
            for y in x:
                if (not(y.colour == colour) and not(y.piece == "Empty")):
                    #print(self.boardPieces[7][4].piece + self.boardPieces[7][4].colour + "OOOOOO")
                    #print(y.piece + " " + y.colour)
                    if(self.IsAttacking(y,kingPiece)):
                        #print(y.piece + " " + y.colour)
                        return True
        return False
        
        
    
    def IsAttacking (self,attacker,defender):
        #print("Attacker: " + attacker.colour + " " + attacker.piece)
        #print("Defender: " + defender.colour + " " + defender.piece)
        curColour = attacker.colour
        #print(attacker.xLocation, attacker.yLocation)
        #print(attacker.piece + " " + attacker.colour)
        #print(defender.piece + " " + defender.colour)
        self.count+= 1
        if (attacker.piece == "Pawn"):
        
            if (curColour == "White"):
                try:     
                    if((defender.xLocation + 1 == attacker.xLocation and defender.yLocation - 1 == attacker.yLocation or defender.xLocation - 1 == attacker.xLocation and defender.yLocation - 1 == attacker.yLocation)):
                        return True
                except:
                    pass
      
            elif(curColour == "Black"):
                 try:  
                    if((defender.xLocation + 1 == attacker.xLocation and defender.yLocation + 1 == attacker.yLocation or defender.xLocation - 1 == attacker.xLocation and defender.yLocation + 1 == attacker.yLocation)):
                        return True
                 except:
                     pass

        elif(attacker.piece == "Knight"):
            try:
                if (abs(defender.xLocation - attacker.xLocation) == 2 and abs(defender.yLocation - attacker.yLocation) == 1 or abs(defender.xLocation - attacker.xLocation) == 1 and abs(defender.yLocation - attacker.yLocation) == 2):
                    
                    #print(attacker.xLocation)
                    #print(attacker.yLocation) 
                    return True
            except:
                pass
                    
                    
        elif(attacker.piece == "Bishop"):
            
            if (abs(defender.xLocation - attacker.xLocation) ==abs(defender.yLocation - attacker.yLocation)):
                    
                xMultiplier = 1
                yMultiplier = 1
                isLegal = True
                
                if(defender.xLocation > attacker.xLocation):
                    xMultiplier = -1
                if (defender.yLocation > attacker.yLocation):
                    yMultiplier = -1
                for square in range(1,abs(defender.xLocation - attacker.xLocation)):
                    
                    if(not(self.boardPieces[defender.xLocation + square*xMultiplier - 1][defender.yLocation + square*yMultiplier - 1].piece == "Empty")):
                        isLegal = False
                if (isLegal):
                    return True
            
        elif(attacker.piece == "Rook"):
            #print(defender.piece + " AAAAA " + defender.colour)
            if(defender.xLocation == attacker.xLocation or defender.yLocation == attacker.yLocation):
                    
                xMultiplier = 1
                yMultiplier = 1
                isLegal = True
                
                if (defender.xLocation == attacker.xLocation):
                    if (defender.yLocation > attacker.yLocation):
                        yMultiplier = -1
                    for square in range(1,abs(defender.yLocation - attacker.yLocation)):
                    
                        if(not(self.boardPieces[defender.xLocation-1][defender.yLocation + square*yMultiplier - 1].piece == "Empty")):
                            isLegal = False
                    
                else:
                    if(defender.xLocation > attacker.xLocation):
                        xMultiplier = -1
                
                    for square in range(1,abs(defender.xLocation - attacker.xLocation)):
                    
                        if(not(self.boardPieces[defender.xLocation + square*xMultiplier - 1][defender.yLocation-1].piece == "Empty")):
                            isLegal = False
                if (isLegal):
                    return True            
            
        elif(attacker.piece == "Queen"):
            
            xMultiplier = 1
            yMultiplier = 1
            isLegal = True
                
                
            if (abs(defender.xLocation - attacker.xLocation) == abs(defender.yLocation - attacker.yLocation)):
                if(defender.xLocation > attacker.xLocation):
                    xMultiplier = -1
                if (defender.yLocation > attacker.yLocation):
                    yMultiplier = -1
                for square in range(1,abs(defender.xLocation - attacker.xLocation)):
                    
                    if(not(self.boardPieces[defender.xLocation + square*xMultiplier - 1][defender.yLocation + square*yMultiplier - 1].piece == "Empty")):
                        isLegal = False
            elif(defender.xLocation == attacker.xLocation or defender.yLocation == attacker.yLocation):
                if (defender.xLocation == attacker.xLocation):
                    if (defender.yLocation > attacker.yLocation):
                        yMultiplier = -1
                    for square in range(1,abs(defender.yLocation - attacker.yLocation)):
                    
                        if(not(self.boardPieces[defender.xLocation-1][defender.yLocation + square*yMultiplier - 1].piece == "Empty")):
                            isLegal = False
                else:
                    if(defender.xLocation > attacker.xLocation):
                        xMultiplier = -1
                
                    for square in range(1,abs(defender.xLocation - attacker.xLocation)):
                    
                        if(not(self.boardPieces[defender.xLocation + square*xMultiplier - 1][defender.yLocation-1].piece == "Empty")):
                            isLegal = False
            else:
                isLegal = False
            if(isLegal):
                return True
            
        elif(attacker.piece == "King"):
            if((abs(defender.xLocation - attacker.xLocation) == 1 and defender.yLocation == attacker.yLocation) or (abs(defender.yLocation - attacker.yLocation) == 1 and defender.xLocation == attacker.xLocation) or (abs(defender.xLocation - attacker.xLocation) == 1 and abs(defender.yLocation - attacker.yLocation) == 1)):
                return True
        return False   

    def isCheckMate(self,colour):
        isLegal = False
        isCapture = False
        moves = []
        boardCopy = copy.deepcopy(self.boardPieces)
        """
        kingPiece = None
        for x in self.boardPieces:
            for y in x:
                if (y.piece == "King" and y.colour == colour):
                    kingPiece = y
        """
        for x in self.boardPieces:
            for y in x:
                if (y.colour == colour and not(y.piece == "Empty")):
                    moves = []
                    self.moveAppender(moves,y,colour)
                    
                    for item in moves:
                        """
                        print("KKKKKKKKKKKKKKKK")
                        print(y.xLocation,y.yLocation)
                        print(self.boardPieces[y.xLocation - 1][y.yLocation - 1].piece)
                        print(y.piece)
                        print(item.xLocation, item.yLocation)
                        """
                        self.pieceToBeMoved = self.boardPieces[y.xLocation - 1][y.yLocation - 1]
                        """
                        print(y.xLocation - 1,y.yLocation - 1)
                        print(self.boardPieces[4][7].piece + self.boardPieces[4][7].colour)
                        print(self.boardPieces[5][7].piece + self.boardPieces[5][7].colour)
                        print(self.boardPieces[item.xLocation - 1][item.yLocation - 1].piece + " " + self.boardPieces[item.xLocation - 1][item.yLocation - 1].colour)
                        print(self.boardPieces[item.xLocation - 1][item.yLocation - 1].xLocation,self.boardPieces[item.xLocation - 1][item.yLocation - 1].yLocation)
                        """
                        isLegal,isCapture = self.isLegalMove(self.boardPieces[item.xLocation - 1][item.yLocation - 1],item.xLocation,item.yLocation,colour)
                        """
                        print("QQQQQQQQQQQQQQQ")
                        for a in self.boardPieces:
                            for b in a:
                                print(b.piece + "  ")
                            print()
                        print("EEEEEEEEEEEEEE")
                        
                        
                        print(y.xLocation,y.yLocation)
                        print(isLegal)
                        print(item.piece)
                        print(item.xLocation, item.yLocation)
                        """
                        if(isLegal):
                            #print(self.boardPieces[kingPiece.xLocation - 1][kingPiece.yLocation - 1].piece + " BBBBB " + self.boardPieces[kingPiece.xLocation - 1][kingPiece.yLocation - 1].colour)
                            self.boardPieces[item.xLocation - 1][item.yLocation - 1] = item
                            #print(self.boardPieces[kingPiece.xLocation - 1][kingPiece.yLocation - 1].piece + " CCCCC " + self.boardPieces[kingPiece.xLocation - 1][kingPiece.yLocation - 1].colour)
                            #print(self.boardPieces[4][7].piece + self.boardPieces[4][7].colour)
                            #print(self.boardPieces[5][7].piece + self.boardPieces[5][7].colour)
                            self.boardPieces[y.xLocation - 1][y.yLocation - 1].piece = "Empty"
                            self.boardPieces[y.xLocation - 1][y.yLocation - 1].colour = "None"
                            
                            #print(y.xLocation,y.yLocation)
                            if(not(self.isInCheck(colour))):
                                """
                                print("QEEEEEE")
                                print(item.piece)
                                print(item.xLocation, item.yLocation)
                                print(self.boardPieces[item.xLocation - 1][item.yLocation - 1].piece + "TTTT" + self.boardPieces[item.xLocation - 1][item.yLocation - 1].colour)
                                print(kingPiece.xLocation,kingPiece.yLocation)
                                print(self.boardPieces[kingPiece.xLocation - 1][kingPiece.yLocation - 1].piece + " CCCCC " + self.boardPieces[kingPiece.xLocation - 1][kingPiece.yLocation - 1].colour)
                                
                                if(item.piece == self.boardPieces[kingPiece.xLocation - 1][kingPiece.yLocation - 1].piece):
                                    print("QQQQQQQQQQQQQQQ")
                                    for a in self.boardPieces:
                                        for b in a:
                                            print(b.piece + "  ")
                                        print()
                                    print("EEEEEEEEEEEEEE")
                                """
                                
                                self.boardPieces = copy.deepcopy(boardCopy)
                                return False
                            self.boardPieces = copy.deepcopy(boardCopy)
                        #print(y.xLocation,y.yLocation)
                    
        return True
    
    def moveAppender(self,moves,y,colour):
        if (y.piece == "Pawn" ):
            
            if (colour == "Black"):
                moves.append(Pieces(y.piece,y.colour,y.xLocation , y.yLocation - 1)) 
                if (y.xLocation >= 1 and y.xLocation <= 7):                            
                    moves.append(Pieces(y.piece,y.colour,y.xLocation + 1, y.yLocation - 1))
                if (y.xLocation >= 2 and y.xLocation <= 8):                            
                    moves.append(Pieces(y.piece,y.colour,y.xLocation - 1, y.yLocation - 1))        
                if (y.yLocation == 7):                            
                    moves.append(Pieces(y.piece,y.colour,y.xLocation, y.yLocation - 2))
                
            else:
                moves.append(Pieces(y.piece,y.colour,y.xLocation , y.yLocation + 1)) 
                if (y.xLocation >= 1 and y.xLocation <= 7):                            
                    moves.append(Pieces(y.piece,y.colour,y.xLocation + 1, y.yLocation + 1))
                if (y.xLocation >= 2 and y.xLocation <= 8):                            
                    moves.append(Pieces(y.piece,y.colour,y.xLocation - 1, y.yLocation + 1))        
                if (y.yLocation == 2):                            
                    moves.append(Pieces(y.piece,y.colour,y.xLocation, y.yLocation + 2))

            
        elif (y.piece == "Knight"):
            
            if (y.xLocation + 2 <= 8 and y.yLocation + 1 <= 8):
                moves.append(Pieces(y.piece,y.colour,y.xLocation + 2, y.yLocation + 1))
            if (y.xLocation + 2 <= 8 and y.yLocation - 1 >= 1):
                moves.append(Pieces(y.piece,y.colour,y.xLocation + 2, y.yLocation - 1))
            if (y.xLocation - 2 >= 1 and y.yLocation + 1 <= 8):
                moves.append(Pieces(y.piece,y.colour,y.xLocation - 2, y.yLocation + 1))
            if (y.xLocation - 2 >= 1 and y.yLocation - 1 >= 1):
                moves.append(Pieces(y.piece,y.colour,y.xLocation - 2, y.yLocation - 1))
            
            if (y.xLocation + 1 <= 8 and y.yLocation + 2 <= 8):
                moves.append(Pieces(y.piece,y.colour,y.xLocation + 1, y.yLocation + 2))
            if (y.xLocation + 1 <= 8 and y.yLocation - 2 >= 1):
                #print("WAZOO")
                moves.append(Pieces(y.piece,y.colour,y.xLocation + 1, y.yLocation - 2))
            if (y.xLocation - 1 >= 1 and y.yLocation + 2 <= 8):
                moves.append(Pieces(y.piece,y.colour,y.xLocation - 1, y.yLocation + 2))
            if (y.xLocation - 1 >= 1 and y.yLocation - 2 >= 1):
                moves.append(Pieces(y.piece,y.colour,y.xLocation - 1, y.yLocation - 2))
            
            
        elif (y.piece == "Bishop"):
            for i in range(1,8):
                if(y.xLocation + i <= 8 and y.yLocation + i <= 8):
                    moves.append(Pieces(y.piece,y.colour,y.xLocation + i, y.yLocation + i))
                if(y.xLocation + i <= 8 and y.yLocation - i >= 1):
                    moves.append(Pieces(y.piece,y.colour,y.xLocation + i, y.yLocation - i))
                if(y.xLocation - i >= 1 and y.yLocation + i <= 8):
                    moves.append(Pieces(y.piece,y.colour,y.xLocation - i, y.yLocation + i))
                if(y.xLocation - i >= 1 and y.yLocation - i >= 1):
                    moves.append(Pieces(y.piece,y.colour,y.xLocation - i, y.yLocation - i))

            
        elif (y.piece == "Rook"):
            
            for i in range(1,8):
                if(y.xLocation + i <= 8):
                    moves.append(Pieces(y.piece,y.colour,y.xLocation + i, y.yLocation))
                if(y.yLocation - i >= 1):
                    moves.append(Pieces(y.piece,y.colour,y.xLocation, y.yLocation - i))
                if(y.xLocation - i >= 1):
                    moves.append(Pieces(y.piece,y.colour,y.xLocation - i, y.yLocation))
                if(y.yLocation + i <= 8):
                    moves.append(Pieces(y.piece,y.colour,y.xLocation, y.yLocation + i))

            
        elif (y.piece == "Queen"):
            
            for i in range(1,8):
                
                if(y.xLocation + i <= 8 and y.yLocation + i <= 8):
                    moves.append(Pieces(y.piece,y.colour,y.xLocation + i, y.yLocation + i))
                if(y.xLocation + i <= 8 and y.yLocation - i >= 1):
                    moves.append(Pieces(y.piece,y.colour,y.xLocation + i, y.yLocation - i))
                if(y.xLocation - i >= 1 and y.yLocation + i <= 8):
                    moves.append(Pieces(y.piece,y.colour,y.xLocation - i, y.yLocation + i))
                if(y.xLocation - i >= 1 and y.yLocation - i >= 1):
                    moves.append(Pieces(y.piece,y.colour,y.xLocation - i, y.yLocation - i))
                
                if(y.xLocation + i <= 8):
                    moves.append(Pieces(y.piece,y.colour,y.xLocation + i, y.yLocation))
                if(y.yLocation - i >= 1):
                    moves.append(Pieces(y.piece,y.colour,y.xLocation, y.yLocation - i))
                if(y.xLocation - i >= 1):
                    moves.append(Pieces(y.piece,y.colour,y.xLocation - i, y.yLocation))
                if(y.yLocation + i <= 8):
                    moves.append(Pieces(y.piece,y.colour,y.xLocation, y.yLocation + i))

            
        elif (y.piece == "King"):
            if (y.xLocation + 1 <= 8):
                moves.append(Pieces(y.piece,y.colour,y.xLocation + 1, y.yLocation))
                if(y.yLocation + 1 <= 8):
                    moves.append(Pieces(y.piece,y.colour,y.xLocation + 1, y.yLocation + 1))
                if(y.yLocation - 1 >= 1):
                    moves.append(Pieces(y.piece,y.colour,y.xLocation + 1, y.yLocation - 1))
            if(y.xLocation - 1 >= 1):
                moves.append(Pieces(y.piece,y.colour,y.xLocation - 1, y.yLocation))
                if(y.yLocation + 1 <= 8):
                    moves.append(Pieces(y.piece,y.colour,y.xLocation - 1, y.yLocation + 1))
                if(y.yLocation - 1 >= 1):
                    moves.append(Pieces(y.piece,y.colour,y.xLocation - 1, y.yLocation - 1))
            if (y.yLocation + 1 <= 8):
                moves.append(Pieces(y.piece,y.colour,y.xLocation, y.yLocation + 1))
            if (y.yLocation - 1 >= 1):
                moves.append(Pieces(y.piece,y.colour,y.xLocation, y.yLocation - 1))
        
board = Layout()
board.drawboard()
#board.test()
board.mainloop()

