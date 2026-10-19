import numpy as np
import chess
import chess.pgn
import tf2onnx
import onnx
import onnxruntime as ort
import platform
import gc
from itertools import zip_longest
from timeit import default_timer as timer
from numba import cuda
import tensorflow as tf
from eval_func import evaluate_board
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler

from ChessAI import ChessAI

gameStart = 21
gameUntil = 41

trainingCount = 0

clip_min = -10
clip_max = 10

# Scale factor to map the clipped range to [-1, 1]
clip_range = clip_max - clip_min
scale_factor = 2 / clip_range  # To map range [-20, 20] to [-1, 1]

if platform.system() == 'Windows':
    data_path1 = '../Models/BlackModel4.keras'
    data_path2 = '../Models/WhiteModel1.keras'
    data_path3 = '../Models/WhiteEval_21_36.keras'
elif platform.system() == 'Linux':
    data_path1 = '/mnt/c/Users/Kumodth/Desktop/Programming/Chess Engine/Chess-Engine/Models/BlackModel4.keras'
    data_path2 = '/mnt/c/Users/Kumodth/Desktop/Programming/Chess Engine/Chess-Engine/Models/WhiteModel1.keras'
    data_path3 = '/mnt/c/Users/Kumodth/Desktop/Programming/Chess Engine/Chess-Engine/Models/WhiteEval_21_36.keras'

blackModel = tf.keras.models.load_model(data_path1)
whiteModel = tf.keras.models.load_model(data_path2)

# Assuming you have models already defined as black_model and white_model
alpha_ux = ChessAI(blackModel, whiteModel, chess.Board())

def clip_and_scale_target(eval_value):
    """Clip the target evaluation score to the range [-20, 20] and then scale to [-1, 1]."""
    clipped_value = np.clip(eval_value / 1000, clip_min, clip_max)
    # print(eval_value, clipped_value)
    return (clipped_value - clip_min) * scale_factor - 1  # Scale to [-1, 1]

def reverse_scaling_and_unclip(scaled_output):
    """Reverse the scaling (to get back to the original centipawn range) and unclip it."""
    unscaled_value = (scaled_output + 1) * (clip_range / 2) + clip_min
    return unscaled_value * 1000

def popcount(bb):
    return bin(bb).count('1')

def game_phase(board: chess.Board):
    piece_num = popcount(board.occupied) - 2
    
    if (board.queens == 0):
        is_endgame = piece_num < 18
        is_near_gameEnd = piece_num < 12
    else:	
        is_endgame = piece_num < 16
        is_near_gameEnd = piece_num < 10
        
    if is_near_gameEnd:
        return 0
    elif is_endgame:
        return 1
    else:
        return 2
    

def encode_board(board: chess.Board) -> np.ndarray:
    """
    Converts a chess board into a flat 771-length numpy array matching C++ encoding.
    """
    encoding = np.zeros((768 + 3,), dtype=np.float32)

    piece_map = board.piece_map()
    for square, piece in piece_map.items():
        row = square // 8
        col = square % 8
        piece_type = piece.piece_type  # 1-6
        color = piece.color  # True = white, False = black

        piece_index = (piece_type - 1) + (6 if not color else 0)
        index = row * 8 * 12 + col * 12 + piece_index
        encoding[index] = 1.0

    phase = game_phase(board)
    encoding[768 + phase] = 1.0

    return encoding

def reflect_board(board):
    
    """
    A function that reflects the board across the verticl axis

    Parameters:
    - board: chess.Board, the current board state

    Returns:
    - The reflected chess.Board object
    """
    
    # Create a new board which is a reflection of the input board
    reflected_board = chess.Board()
    reflected_board.clear()  # Clear the board first
    
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            # Calculate the reflected square
            file = chess.square_file(square)
            rank = chess.square_rank(square)
            reflected_file = 7 - file
            reflected_square = chess.square(reflected_file, rank)
            reflected_board.set_piece_at(reflected_square, piece)
    
    return reflected_board

def generalTraining():
    
    """
    Function to acquire moves from past games
    """ 
    
    print ("Loading general training data...")

    inputData = []
    output = []
    
    global gameStart
    global gameUntil
    
    
    # Open game data file
    if platform.system() == 'Windows':
        data_path = r'../PGNs/SuperSet.pgn'
    elif platform.system() == 'Linux':
        data_path = '/mnt/c/Users/Kumodth/Desktop/Programming/Chess Engine/Chess-Engine/PGNs/LargeSet.pgn'  # Example for WSL
    pgn = open(data_path)
    
    # Holds the starting position for training
    gamePosition = gameStart
    
    # Iterate through all moves and play them on a board.
    count = 1
    while True:
        # print(count)
        # Exit the loop when the game file has reached its end
        game = chess.pgn.read_game(pgn)
        if game is None:            
            break
        
        # Set the game board to the starting position and move count to 1
        board = game.board()
        inGameCount = 1
        for main_line_move in game.mainline_moves():
            
            # Check if the game count is within the training bounds
            if (inGameCount >= gamePosition and inGameCount <= gameUntil):
                
                if (inGameCount % 2 == 0):
                    
                    moves, evaluations = alpha_ux.create_test_data(board)
                    
                    for move, evaluation in zip(moves,evaluations):
                        board.push(move)
                        
                        inputData.append(encode_board(board))
                        inputData.append(encode_board(reflect_board(board))) 

                        result = clip_and_scale_target(evaluation)
        
                        output.append(result)
                        output.append(result)
                        board.pop()
                
                        count += 1
                
            # Make the move
            board.push(main_line_move)    
                
            # Increment the counts             
            inGameCount += 1
                
    print ("Loaded: ", count * 2, " positions")
    return inputData, output

def captureTraining():
    
    """
    Function to acquire capture moves
    """ 
    
    print ("Loading capture training data...")
    
    inputData = []
    output = []
    
    global gameStart
    global gameUntil
    
    # Open game data file
    if platform.system() == 'Windows':
        data_path = r'../PGNs/SuperSet.pgn'
    elif platform.system() == 'Linux':
        data_path = '/mnt/c/Users/Kumodth/Desktop/Programming/Chess Engine/Chess-Engine/PGNs/testgames8.pgn'
    pgn = open(data_path)
    
    # Holds the starting position for training
    gamePosition = gameStart
    
    # Iterate through all moves and play them on a board.
    count = 1
    
    while True:
        
        # Exit the loop when the game file has reached its end
        game = chess.pgn.read_game(pgn)
        if game is None:            
            break
        
        # Set the game board to the starting position and move count to 1
        board = game.board()
        
        inGameCount = 1        
        for move in game.mainline_moves():
            
            board.push(move)
    
            # Check if the game count is within the training bounds
            if (inGameCount >= gamePosition and inGameCount <= gameUntil):
                
                board.pop()
                if board.is_capture(move):
                    
                    inputData.append(encode_board(board)) 
                    inputData.append(encode_board(reflect_board(board)))
                    
                    result = clip_and_scale_target(evaluate_board (board))
                    
                    output.append(result)
                    output.append(result)
                    
                    count += 1
                board.push(move)
                
            # Increment the counts and make the move
               
            inGameCount += 1        
    
    print ("Loaded: ", count, " positions")
    return inputData, output
  
def checkmateTraining():
    
    """
    Function to acquire checkmate moves
    """ 
    
    print ("Loading checkmate training data...")
    
    inputData = []
    output = []
    
    global gameStart
    global gameUntil
    
    # Open game data file
    if platform.system() == 'Windows':
        data_path = r'../PGNs/blackWins.pgn'
    elif platform.system() == 'Linux':
        data_path = '/mnt/c/Users/Kumodth/Desktop/Programming/Chess Engine/Chess-Engine/PGNs/testgames8.pgn'
    pgn = open(data_path)
    gamePosition = 1
    count = 1
    while True:
        
        # Exit the loop when the game file has reached its end
        game = chess.pgn.read_game(pgn)
        if game is None:            
            break
        
        # Set the game board to the starting position and move count to 1
        board = game.board()
        
        inGameCount = 1
        for move in game.mainline_moves():
            board.push(move)
            # Check if the game count is within the training bounds
            if (inGameCount >= gamePosition and inGameCount <= 300):
                
                if board.is_checkmate():
                    
                    board.pop() 
                    inputData.append(encode_board(board)) 
                    inputData.append(encode_board(reflect_board(board)))
                    
                    result = clip_and_scale_target(evaluate_board (board))
                    
                    output.append(result)
                    output.append(result)
                    board.push(move)
                    
                    count += 1   
            # Increment the counts and make the move
            
            inGameCount += 1
    
    print ("Loaded: ", count, " positions")            
    return inputData, output 

def evasionTraining():
    
    """
    Function to acquire evasive moves
    """ 
    
    print ("Loading evasion training data...")
        
    inputData = []
    output = []
    
    global gameStart
    global gameUntil
    
    # Open game data file
    if platform.system() == 'Windows':
        data_path = r'../PGNs/blackWins.pgn'
    elif platform.system() == 'Linux':
        data_path = '/mnt/c/Users/Kumodth/Desktop/Programming/Chess Engine/Chess-Engine/PGNs/LargeSet.pgn'
    pgn = open(data_path)
    
    gamePosition = gameStart
    
    # Iterate through all moves and play them on a board.
    count = 1
    while True:
        
        # Exit the loop when the game file has reached its end
        game = chess.pgn.read_game(pgn)
        if game is None:            
            break
        
        # Set the game board to the starting position and move count to 1
        board = game.board()
        
        inGameCount = 1
        for move in game.mainline_moves():
    
            # Check if the game count is within the training bounds
            if (inGameCount >= gamePosition and inGameCount <= gameUntil):
                
                    
                for legalMove in board.legal_moves:
                    from_square = legalMove.from_square
                    to_square = legalMove.to_square
                    piece_under_attack = board.is_attacked_by(not board.turn, from_square)

                    if piece_under_attack:
                        
                        board.push(legalMove)
                        # Check if the destination square is not attacked by the opponent
                        if not board.is_attacked_by(board.turn, to_square):
                            
                            inputData.append(encode_board(board)) 
                            inputData.append(encode_board(reflect_board(board)))
                            
                            result = clip_and_scale_target(evaluate_board (board))
                            
                            output.append(result)
                            output.append(result)
                            
                            count += 1
                        board.pop()
                
            # Increment the counts and make the move
               
            inGameCount += 1
            board.push(move)
            
    print ("Loaded: ", count, " positions")        
    return inputData, output


def create_nnue_model(input_shape=(768 + 3,)):
    """
    Create a simplified NNUE model using fully connected layers, without complex convolutions,
    making it easier to port to C++ later.
    """
    inputs = layers.Input(shape=input_shape)
    
    # Dense layers for evaluation, avoiding convolution and batch norm complexities
    x = layers.Dense(512, activation='relu', kernel_initializer='he_uniform')(inputs)
    x = layers.Dense(256, activation='relu', kernel_initializer='he_uniform')(x)
    x = layers.Dense(128, activation='relu', kernel_initializer='he_uniform')(x)
    
    # Output layer to predict evaluation score
    outputs = layers.Dense(1)(x)

    # Create and compile the model
    model = models.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    return model

def get_data():
    # Reset the GPU
    cuda.select_device(0)
    cuda.current_context().reset()
    
    inputData, output = generalTraining()    
    
    # # Acquire evasion data and interleave the data
    # a,b = evasionTraining()    
    # inputData = [item for pair in zip_longest(inputData, a) for item in pair if item is not None]
    # output = [item for pair in zip_longest(output, b) for item in pair if item is not None]
    
    # # Remove the unused list data to save space
    # del a,b
    # gc.collect()
    
    # # Acquire capture data and interleave the data
    # a,b = captureTraining()    
    # inputData = [item for pair in zip_longest(inputData, a) for item in pair if item is not None]
    # output = [item for pair in zip_longest(output, b) for item in pair if item is not None]
    
    # # Remove the unused list data to save space
    # del a,b
    # gc.collect()
    
    # # Acquire checkmate data and interleave the data
    # a,b = checkmateTraining()    
    # inputData = [item for pair in zip_longest(inputData, a) for item in pair if item is not None]
    # output = [item for pair in zip_longest(output, b) for item in pair if item is not None]
    
    # # Remove the unused list data to save space
    # del a,b
    # gc.collect()
    
    return inputData,output

# def lr_schedule(epoch, lr):
#     """
#     Custom LR schedule:
#     - Linearly reduce initial LR for warm start
#     - Halve every N epochs
#     - Never go below min_lr
#     """
#     decay_factor = 0.5
#     decay_every = 3
#     min_lr = 2e-4

#     initial_lr = 0.0005
#     warmup_offset = trainingCount * 0.000005
#     lr = initial_lr - warmup_offset

#     if epoch != 0 and epoch % decay_every == 0:
#         lr = max(lr * decay_factor, min_lr)
    
#     return lr

def lr_schedule(epoch, training_count, initial_lr=5e-4, decay_factor=0.8, decay_every=3, min_lr=1e-5):
    """
    Learning rate scheduler function that adjusts learning rate based on both training count and epoch.
    
    Parameters:
    - epoch: The current epoch within the current training block.
    - training_count: A global count that tracks how many training blocks have been executed.
    - initial_lr: The starting learning rate.
    - decay_factor: Factor by which the learning rate decays after each block.
    - decay_every: Number of epochs after which the learning rate should decay within a block.
    - min_lr: Minimum allowable learning rate.
    
    Returns:
    - The updated learning rate.
    """
    
    # Decrease the base learning rate after each training block
    base_lr = initial_lr * (decay_factor ** training_count)
    
    # Now apply the finer decay based on epochs within the block
    lr = base_lr * (decay_factor ** (epoch // decay_every))
    
    # Ensure the learning rate does not go below the minimum threshold
    return max(lr, min_lr)

def train_model(model):
    
    t0 = timer()
    global trainingCount
    inputs,outputs = get_data()
    
    num_samples = len(inputs)
    print("Input Size: ", len(inputs))
    
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    # ** Training loop **
    
    # Iterate through entire training data multiple times
    for i in range (5):
        print ("Iteration:", i)
        
        # Loop through multiple partitions to not fill the GPU
        for start_idx in range(0, num_samples, 100000):
            end_idx = min(start_idx + 100000, num_samples)
            
            # Convert the input and output into numpy arrays
            x = np.array(inputs[start_idx:end_idx])
            y = np.array(outputs[start_idx:end_idx])
            
            # Reset the GPU
            cuda.select_device(0)
            cuda.current_context().reset()
            
            print("Starting Batch:",trainingCount+1, "From index:",start_idx, "to:", end_idx,'\n')
            
            # Compile the model and train on the given data
            lr_scheduler = LearningRateScheduler(lr_schedule)            
            
            # Implement Early Stopping
            early_stopping = EarlyStopping(
                monitor='val_mae',  # Metric to monitor
                patience=5,          # Number of epochs with no improvement after which training will be stopped
                restore_best_weights=True  # Restore the model weights from the epoch with the best value of the monitored quantity
            )
            
            # Train the model with early stopping
            history = model.fit(
                x, y,
                epochs=500,  # Set a large number of epochs for the possibility of early stopping
                batch_size=64,
                validation_split=0.2,  # Split a portion of the data for validation
                callbacks=[lr_scheduler, early_stopping],  # Pass the early stopping callback and learning rate scheduler
                shuffle=True,
                verbose=1
                )
            trainingCount +=1
            print("Done:",trainingCount)
            
            # Delete the partition and call the garbage collector
            del x, y
            gc.collect()            
            
    print(model.summary())
    t1 = timer()
    print("Time elapsed: ", t1 - t0)
    return model

model = create_nnue_model()
# a,b = get_data()
# print (a,b)
model = train_model(model)

# # Assuming 'model' is your trained model
# onnx_model = tf2onnx.convert.from_keras(model)
# onnx.save_model(onnx_model, "nnue_model.onnx")

# Create a tf.function wrapper with input signature
input_signature = [tf.TensorSpec([None, 771], tf.float32, name="input")]

@tf.function(input_signature=input_signature)
def model_func(x):
    return model(x)

onnx_model, _ = tf2onnx.convert.from_function(
    model_func,
    input_signature=input_signature,
    opset=13,
    output_path="NNUE_treesearch_21_to_61.onnx"
)

if platform.system() == 'Windows':
    data_path = r'NNUE.keras'
elif platform.system() == 'Linux':
    data_path = '/mnt/c/Users/Kumodth/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine/NNUE_treesearch_21_to_61.keras'  # Example for WSL
model.save(data_path)


# Create dummy input data
board = chess.Board("2q1k2r/Q3b1np/R1p1p1p1/2Ppn3/1P1N4/4P3/5PPP/6K1 b k - 0 22")
input_data = np.array([encode_board(board)], dtype=np.float32)

# Run Keras prediction
keras_output = model.predict(input_data)

# Run ONNX prediction
sess = ort.InferenceSession("NNUE_treesearch_21_to_61.onnx")
onnx_output = sess.run(None, {"input": input_data})

print("Keras output:", reverse_scaling_and_unclip(keras_output[0][0]))
print("ONNX output:", onnx_output[0][0][0])
print ('Actual Eval:', evaluate_board (board))

