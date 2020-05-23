"""
import GUI as gui


class Game():
    
    def __init__(self):
        self.board = gui.Layout()
        self.board.drawboard()
        print (self.board.boardPieces[0][0].piece)
        self.board.mainloop()
        
    

game = Game()
"""