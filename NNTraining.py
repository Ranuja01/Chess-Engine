import numpy as np
import chess
import chess.pgn
from pickle import load
from pickle import dump
from timeit import default_timer as timer

board = chess.Board()
board.legal_moves
model = load(open('OpeningModel.pkl', 'rb'))


def redesignBoard(board):
    newBoard = [[]]
    for i in range(112,127,2):
        for j in range(0,8):
            #print(board[i - j * 16], end =" ")
            newBoard[0].append(board[i - j * 16])
        #print()
    return newBoard

def predictionInfo(prediction):
    #print("ASDASDASD")
    pieceToBeMoved = (prediction // 64) + 1
    #print(pieceToBeMoved)    
    pieceToBeMovedXLocation = (pieceToBeMoved // 8) + 1    
    pieceToBeMovedYLocation = pieceToBeMoved % 8
    
    
    
    
    remainder = prediction % 64
    #print(remainder)
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



test = [[0]*4096]
test1 = []
test2 = []

test3 = [

         ['R','P','.','.','.','.','p','r',
         'N','P','.','.','.','.','p','n',
         'B','P','.','.','.','.','p','b',
         'Q','P','.','.','.','.','p','q',
         'K','.','.','P','.','.','p','k',
         'B','P','.','.','.','.','p','b',
         'N','P','.','.','.','.','p','n',
         'R','P','.','.','.','.','p','r']
        
        ]

pgn = open("testgames6.pgn")
t0= timer()
#first_game = chess.pgn.read_game(pgn)
#second_game = chess.pgn.read_game(pgn)
gamePosition = 14
# Iterate through all moves and play them on a board.
count = 1
while True:
    game = chess.pgn.read_game(pgn)
    if game is None:
        
        break
    #print("ASDASD")
    board = game.board()
    inGameCount = 1
    for move in game.mainline_moves():
        
        board.push(move)
        #print(board)
        #print()
        
        strBoard = str(board)
        #print(board.fullmove_number)
        #print(redesignBoard(strBoard))
        #print(board.peek())
        
        if (count >= 1328 and count <= 1510):
            '''
            print(count)
            print(inGameCount)
            print()
            '''
        if (inGameCount <= gamePosition):
            
            if (count % 2 == 1):
                test2.append(redesignBoard(strBoard))
            else:
                
                moveMade = str(board.peek())
                a = ord(moveMade[0:1]) - 96
                b = int(moveMade[1:2])
                c = ord(moveMade[2:3]) - 96
                d = int(moveMade[3:4])
                
                test [0][reversePrediction(a,b,c,d) - 1] = 1
                test1.append(test) 
                test = [[0]*4096]
        count += 1   
        inGameCount += 1
        #print(a,b,c,d)
        #print()
    else:
        #print()  
        if (count % 2 == 0):
            count -= 1
            if(inGameCount <= gamePosition and inGameCount % 2 == 0):
                test2.pop()
                
        if (count < 6000):
            '''
            print(inGameCount)
            print(len(test1))
            print(len(test2))
            print(count)
            print()
            '''
    #print(count)
    

print(count)
convertToAscii(test2)
convertToAscii([test3])

x = np.array([i for i in test2])
y = np.array([i for i in test1],np.int8)

model.compile(optimizer="RMSprop", loss="mse", metrics=["acc"])
for layer in model.layers:
    print(layer.output_shape)
model.fit(x, y, epochs=200, batch_size=32, shuffle = True, verbose=0)
#print(scale_x.inverse_transform(x))
#print(scale_y.inverse_transform(y))
#printBoard(reverse(np.array([test1])))
#print()
q = model.predict(np.array([test3]))
#printBoard(reverse(q))

print(np.argmax(q))

a,b,c,d = predictionInfo(np.argmax(q) + 1)
print(q[0][0][np.argmax(q)]*100,"%")
print("X1: ",a)
print("Y1: ",b)
print("X2: ",c)
print("Y2: ",d)

print(reversePrediction(a,b,c,d))

dump(model, open('OpeningModel.pkl', 'wb'))

t1 = timer()
print("Time elapsed: ", t1 - t0)