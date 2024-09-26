import multiprocessing
import chess
from timeit import default_timer as timer
import time
# Function to process moves and append results to the shared queue
def test1(n, q,q2): 
    board = chess.Board("rn1qkb1r/pb3p2/2p1pn1p/1p2P1p1/2pP4/2N2NB1/PP2BPPP/R2QK2R b KQkq - 0 10")
    print(n)
    if n == 1:
        for move in board.generate_legal_moves(board.pawns):
            board.push(move)
            for move2 in board.generate_legal_moves():
                board.push(move2)
                for move3 in board.generate_legal_moves():
                    board.push(move3)
                    result = n
                    q.put(result)  # Add result to the queue
                    board.pop()
                    # time.sleep(2)
                board.pop()
            board.pop()
    elif (n == 2):
        for move in board.generate_legal_moves(board.knights):
            board.push(move)
            for move2 in board.generate_legal_moves():
                board.push(move2)
                for move3 in board.generate_legal_moves():
                    board.push(move3)
                    result = n
                    q.put(result)  # Add result to the queue
                    board.pop()
                    # time.sleep(2)
                board.pop()
            board.pop()
    elif (n == 3):
        for move in board.generate_legal_moves(board.bishops):
            board.push(move)
            for move2 in board.generate_legal_moves():
                board.push(move2)
                for move3 in board.generate_legal_moves():
                    board.push(move3)
                    result = n
                    q.put(result)  # Add result to the queue
                    board.pop()
                    # time.sleep(2)
                board.pop()
            board.pop()
    elif (n == 4):
        for move in board.generate_legal_moves(board.rooks):
            board.push(move)
            for move2 in board.generate_legal_moves():
                board.push(move2)
                for move3 in board.generate_legal_moves():
                    board.push(move3)
                    result = n
                    q.put(result)  # Add result to the queue
                    board.pop()
                    # time.sleep(2)
                board.pop()
            board.pop()
    elif (n == 5):
        for move in board.generate_legal_moves(board.queens):
            board.push(move)
            for move2 in board.generate_legal_moves():
                board.push(move2)
                for move3 in board.generate_legal_moves():
                    board.push(move3)
                    result = n
                    q.put(result)  # Add result to the queue
                    board.pop()
                    # time.sleep(2)
                board.pop()
            board.pop()
    elif (n == 6):
        for move in board.generate_legal_moves(board.kings):
            board.push(move)
            for move2 in board.generate_legal_moves():
                board.push(move2)
                for move3 in board.generate_legal_moves():
                    board.push(move3)
                    result = n
                    q.put(result)  # Add result to the queue
                    board.pop()
                    # time.sleep(2)
                board.pop()
            board.pop()
    elif(n == 7):
        while(True):            
            if(not q.empty()):
                q2.put(q.get())
        # test3(q2,q)
            
# Function to process moves and append results to the shared queue
def test2(q): 
    board = chess.Board("rn1qkb1r/pb3p2/2p1pn1p/1p2P1p1/2pP4/2N2NB1/PP2BPPP/R2QK2R b KQkq - 0 10")
    print("SSS")
    for move in board.generate_legal_moves():
        board.push(move)
        for move2 in board.generate_legal_moves():
            board.push(move2)
            for move3 in board.generate_legal_moves():
                board.push(move3)
                result = 1
                q.put(result)  # Add result to the queue
                board.pop()
            board.pop()
        board.pop()

def test3(q2,q): 
    
    while(True):            
        if(not q.empty()):
            q2.put(q.get())
        

if __name__ == "__main__": 
    # Input list
    mylist = [1,2,3,4,5,6,7] 
    count = 0
    nonPawnCount = 0
    t1 = 0
    
    # Use Manager to create a shared Queue
    # with multiprocessing.Manager() as manager:
    manager = multiprocessing.Manager()
    q = manager.Queue()
    q2 = manager.Queue()
    
    t0= timer()
    # Create a pool of workers and map the tasks
    # with multiprocessing.Pool() as pool:
    pool = multiprocessing.Pool()
        
    pool.starmap_async(test1, [(n, q,q2) for n in mylist])
    time.sleep(1)
    
    # pool.starmap_async(test3, (q2,q))
    
    # Run test3 asynchronously, processing the queue concurrently
    
    # pool.starmap_async(test3, [(q2,q)])
    # # Collect results from the queue
    print(q2.qsize())
    print(q.qsize())
    t1 = timer()
    while not q.empty():
        result = q.get()
        count += 1
        t1 = timer()
        
        if (t1 - t0) > 10:
            pool.terminate()  # Terminate the pool
            pool.join()       # Clean up
            break
    print(q2.qsize())
    print(q.qsize())
    print("Time elapsed: ", t1 - t0)
    print(123)
        
    t1 = timer() 
    
    print("Time elapsed: ", t1 - t0)
    
    # print(q2.qsize())
    # time.sleep(2)
    # pool.terminate()
    # pool.join()
    print(count)
    
    
    # count = 0
    # with multiprocessing.Manager() as manager:
    #     q = manager.Queue()
    #     t0= timer()
    #     # Create a pool of workers and map the tasks
    #     with multiprocessing.Pool() as pool:
    #         pool.starmap(test2, [(q,)])

    #     # Collect results from the queue
    #     while not q.empty():
    #         result = q.get()
    #         count += 1
    #         t1 = timer()
            
    #         if (t1 - t0) > 5:
    #             pool.terminate()  # Terminate the pool
    #             pool.join()       # Clean up
    #             break
    #     print("Time elapsed: ", t1 - t0)
    # print(count)