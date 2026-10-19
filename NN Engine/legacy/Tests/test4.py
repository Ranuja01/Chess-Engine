import time
import chess
import chesslib


def is_promotion_move_enhanced(move, board, y):
    
    # Check if the move is a promotion
    if move.promotion is not None:
        return True
    # Check if the move is to the promotion rank
    if (board.turn and y - 1 == 7) or \
       (not(board.turn) and y - 1 == 0):
        # Check if a pawn is moving to the last rank
        from_square = move.from_square
        piece = board.piece_at(from_square)

        if piece and piece.piece_type == 1:
           
            return True
    
    return False

def benchmark_python_chess():
    board = chess.Board()
    
    for i in range (1000):
    
        # Measure time to generate all legal moves
        start_time = time.time()
        for move in board.legal_moves:
            cur = move.uci()
            d = int(cur[3])
            
            
            
            board.push(move)
            
            for square in chess.SQUARES:
                piece = board.piece_at(square)
                if (piece):
                    print(piece, piece.color)
                y = square >> 3
                x = square - (y << 3)
                    
                
                if (x == 3 and y == 3 or
                    x == 3 and y == 4 or
                    x == 4 and y == 3 or
                    x == 4 and y == 4):
                    pass
            
            board.pop()
            
            
        end_time = time.time()
        #print(f"python-chess single set Move Time: {end_time - start_time:.6f} seconds")



def benchmark_python_chess2():
    board = chess.Board()
    
    
    for i in range (1000):
    
        # Measure time to generate all legal moves
        start_time = time.time()
        for move in board.legal_moves:
            cur = move.uci()
            d = int(cur[3])
            
            
            
            board.push(move)
            
            for square in chess.SQUARES:
                piece = board.piece_at(square)
                
                #x = chess.square_file(square)
                y = square >> 3
                x = square - (y << 3)
                    
                
                if square == 27 or square == 35 or square == 26 or square == 36:
                    pass
                
            board.pop()
            
            
        end_time = time.time()
        #print(f"python-chess single set Move Time: {end_time - start_time:.6f} seconds")



# Run benchmarks
print("Benchmarking python-chess...")

start_time = time.time()
benchmark_python_chess()
end_time = time.time()
print(f"python-chess FULL set Move Time: {end_time - start_time:.6f} seconds")

start_time = time.time()
benchmark_python_chess2()
end_time = time.time()
print(f"python-chess2 FULL set Move Time: {end_time - start_time:.6f} seconds")
