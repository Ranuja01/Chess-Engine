# cython_sum.pyx

from libc.stdlib cimport rand, srand, RAND_MAX, malloc, free
from libc.math cimport fmod
from libc.time cimport time
from time import time as py_time 
cimport cython


# # Declare pthread types and functions
# cdef extern from "pthread.h":
#     ctypedef struct pthread_t:
#         pass
#     ctypedef struct pthread_attr_t:
#         pass
#     cdef int pthread_create(pthread_t* thread, const pthread_attr_t* attr, void* (*start_routine)(void*), void* arg)
#     cdef int pthread_join(pthread_t thread, void** retval)
#     cdef int pthread_attr_init(pthread_attr_t* attr)
#     cdef int pthread_attr_destroy(pthread_attr_t* attr)
    


# # Define a struct to hold thread arguments
# cdef struct ThreadArgs:
#     double* numbers
#     int start
#     int end
#     double result

# # Thread function to compute sum of a portion of the list
# cdef void* thread_sum(void* arg):
#     cdef ThreadArgs* args = <ThreadArgs*>arg
#     cdef double total = 0
#     cdef int i
#     for i in range(args.start, args.end):
#         total += args.numbers[i]
#     args.result = total
#     return NULL

# # Function to sum a portion of the list of numbers
# cdef double partial_sum(double[:] numbers, int start, int end):
#     cdef double total = 0
#     cdef int i
#     for i in range(start, end):
#         total += numbers[i]
#     return total

# # Single-threaded summation
# def single_thread_sum(double[:] numbers):
#     cdef double total = partial_sum(numbers, 0, len(numbers))
#     return total

# # Multi-threaded summation using pthreads
# def multi_thread_sum(double[:] numbers, int num_threads):
#     cdef int chunk_size = len(numbers) // num_threads
#     cdef double total = 0
#     cdef pthread_t* threads = <pthread_t*>malloc(num_threads * cython.sizeof(pthread_t))
#     cdef ThreadArgs* thread_args = <ThreadArgs*>malloc(num_threads * cython.sizeof(ThreadArgs))
#     cdef pthread_attr_t attr
#     cdef int i

#     cdef int start
#     cdef int end
#     pthread_attr_init(&attr)

#     for i in range(num_threads):
#         start = i * chunk_size
#         end = start + chunk_size if i != num_threads - 1 else len(numbers)
        
#         # Convert memory view to C-style pointer
#         thread_args[i].numbers = <double*>numbers.data
#         thread_args[i].start = start
#         thread_args[i].end = end
#         thread_args[i].result = 0
        
#         # Create threads
#         pthread_create(&threads[i], &attr, <void*(*)(void*)>thread_sum, &thread_args[i])

#     for i in range(num_threads):
#         pthread_join(threads[i], NULL)
#         total += thread_args[i].result

#     free(threads)
#     free(thread_args)
#     return total

# # Function to generate random numbers
# def generate_numbers(int n):
#     cdef double[:] numbers = cython.view.array(shape=(n,), itemsize=cython.sizeof(cython.double), format="d")
#     cdef int i
#     srand(<uint64_t>time(NULL))
#     for i in range(n):
#         numbers[i] = rand() / RAND_MAX  # Generate a float between 0.0 and 1.0
#         print(numbers[i])
#     return numbers

# # Timing functions
# def time_experiment(int num_numbers, int num_threads):
#     cdef double[:] numbers = generate_numbers(num_numbers)
    
#     cdef double start_time, end_time
    
#     # Single-threaded timing
#     start_time = py_time()
#     total_sum_single = single_thread_sum(numbers)
#     end_time = py_time()
#     single_thread_time = end_time - start_time
    
#     # Multi-threaded timing
#     start_time = py_time()
#     total_sum_multi = multi_thread_sum(numbers, num_threads)
#     end_time = py_time()
#     multi_thread_time = end_time - start_time
    
#     return total_sum_single, single_thread_time, total_sum_multi, multi_thread_time


# Cython_Chess.pyx
from libcpp.vector cimport vector
cdef extern from "stdint.h":
    ctypedef signed char int8_t
    ctypedef unsigned char uint8_t
    ctypedef unsigned long long uint64_t
    

cdef extern from "cpp_bitboard.h":
    void process_bitboards_wrapper(uint64_t * bitboards, int size)
    vector[int] find_most_significant_bits(uint64_t bitmask)
    vector[uint8_t] scan_reversed(uint64_t bb)
    vector[uint8_t] scan_forward(uint64_t bb)
    int attackingScore(int layer[8][8], vector[uint8_t] scanSquares, uint8_t pieceType)
    bint isCapture(uint64_t bb1, uint64_t bb2)
    vector[uint8_t] scan_forward_backward_multithreaded(uint64_t bb)

cdef int layer[2][8][8]
# Initialize layer
# Initialize layer
for i in range(2):
    for j in range(8):
        for k in range(8):
            if i == 0:
                layer[i][j][k] = [
                    [0,0,0,0,0,0,0,0],
                    [0,0,3,3,4,5,5,0],
                    [0,0,3,6,7,6,4,0],
                    [0,0,3,7,8,8,5,0],
                    [0,0,3,7,8,8,5,0],
                    [0,0,3,6,7,6,4,0],
                    [0,0,3,3,4,5,5,0],
                    [0,0,0,0,0,0,0,0]
                ][j][k]
            else:
                layer[i][j][k] = [
                    [0,0,0,0,0,0,0,0],
                    [0,5,5,4,3,3,0,0],
                    [0,4,6,7,6,3,0,0],
                    [0,5,8,8,7,3,0,0],
                    [0,5,8,8,7,3,0,0],
                    [0,4,6,7,6,3,0,0],
                    [0,5,5,4,3,3,0,0],
                    [0,0,0,0,0,0,0,0]
                ][j][k]


cimport cython

def call_process_bitboards(list bitboards):
    cdef int size = len(bitboards)
    cdef uint64_t* c_bitboards = <uint64_t*> malloc(size * sizeof(uint64_t))
    
    
    # Copy data from Python list to C array
    for i in range(size):
        c_bitboards[i] = bitboards[i]
    cdef vector[int] vec
    cdef list result = []
    # Call the C++ function via the wrapper
    process_bitboards_wrapper(c_bitboards, size)
    
    for i in bitboards:
        print("Bitboard:\n ", i)
        print()
        vec = find_most_significant_bits(i)
        size = vec.size()
        result = []
        for i in range(size):
            result.append(vec[i])
        print(result)

def yield_msb(uint64_t bitboard):
    start_time = py_time()
    cdef uint8_t size
    
    cdef vector[uint8_t] vec
        
    vec = scan_forward(bitboard)
    size = vec.size()
    
    for i in range(size):
        # print(vec[i])
        pass
    end_time = py_time()
    return end_time - start_time
                
def test1(board):
    vec = scan_reversed(board.attacks_mask(2))
    size = vec.size()
    total = 0
    for i in range(size):        
    #for attack in chess.scan_reversed(board.attacks_mask(square)): 
        y = vec[i] // 8
        x = vec[i] % 8
        if (board.piece_type_at(2) == 1 or board.piece_type_at(2) == 5):
            total += layer[0][x][y] >> 2
        else:    
            total += layer[0][x][y]        
    print(total)
    
def test2(board):
    vec = scan_forward_backward_multithreaded(board.attacks_mask(2))
    size = vec.size()
    total = 0
    for i in range(size):        
    #for attack in chess.scan_reversed(board.attacks_mask(square)): 
        y = vec[i] // 8
        x = vec[i] % 8
        if (board.piece_type_at(2) == 1 or board.piece_type_at(2) == 5):
            total += layer[0][x][y] >> 2
        else:    
            total += layer[0][x][y]    
    print(total)
def is_capture(board, move):
    for capture in board.generate_legal_captures():
        if (move == capture):
            return True
    return False