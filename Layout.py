# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 15:05:28 2024

@author: Ranuja
"""
import tkinter as tk
from PIL import ImageTk, Image
import easygui

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
        self.n = n
        self.leftframe = tk.Frame(self)
        self.leftframe.grid(row=0, column=0, rowspan=10, padx=40)
        self.middleframe = tk.Frame(self)
        self.middleframe.grid(row=0, column=8, rowspan=8)
        self.canvas = tk.Canvas(self, width=950, height=768, )
        self.canvas.grid(row=0, column=1, columnspan=8, rowspan=8)
        self.board = [[None for row in range(n)] for col in range(n)]

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
