import numpy as np
import chess

def encode_board(board: chess.Board) -> np.ndarray:
    """
    Converts a chess board into a 8x8x12 numpy array representing the pieces.
    """
    encoding = np.zeros((8, 8, 12), dtype=np.float32)

    # Piece map from python-chess
    piece_map = board.piece_map()

    for square, piece in piece_map.items():
        row, col = divmod(square, 8)  # Convert square to row, col
        piece_type = piece.piece_type  # 1: Pawn, 2: Knight, 3: Bishop, etc.
        color = piece.color  # True for white, False for black

        # Mapping piece types and colors to the encoding
        piece_index = (piece_type - 1) + (6 if color == chess.BLACK else 0)
        
        # Set the corresponding entry to 1
        encoding[row, col, piece_index] = 1
    
    return encoding

import tensorflow as tf
from tensorflow.keras import layers, models

def create_simple_nnue_model(input_shape=(8, 8, 12)):
    """
    Create a simplified NNUE model using fully connected layers, without complex convolutions,
    making it easier to port to C++ later.
    """
    inputs = layers.Input(shape=input_shape)

    # Flatten the input data into a 1D vector
    x = layers.Flatten()(inputs)
    
    # Dense layers for evaluation, avoiding convolution and batch norm complexities
    x = layers.Dense(256, activation='relu', kernel_initializer='he_uniform')(x)
    x = layers.Dense(128, activation='relu', kernel_initializer='he_uniform')(x)
    x = layers.Dense(64, activation='relu', kernel_initializer='he_uniform')(x)
    
    # Output layer to predict evaluation score
    outputs = layers.Dense(1)(x)

    # Create and compile the model
    model = models.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

    return model

import onnx
import tf2onnx
import tensorflow as tf

# Assuming 'model' is your trained model
onnx_model = tf2onnx.convert.from_keras(model)
onnx.save_model(onnx_model, "nnue_model.onnx")
