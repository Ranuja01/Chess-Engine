# -*- coding: utf-8 -*-
"""
Created on Fri May  9 15:33:29 2025

@author: Kumodth
"""
from eval_func import evaluate_board
import chess
from timeit import default_timer as timer
import tf2onnx
import onnx
import onnxruntime as ort
import numpy as np

clip_min = -10
clip_max = 10

# Scale factor to map the clipped range to [-1, 1]
clip_range = clip_max - clip_min
scale_factor = 2 / clip_range  # To map range [-20, 20] to [-1, 1]

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


# board1 = chess.Board("8/4P3/7p/5BpP/2p3P1/1k3b2/2p5/2K5 b - - 0 61")
# board2 = chess.Board("r1bqkb1r/p2n1ppp/2p1pn2/1p6/3P4/1BN1PN2/PP3PPP/R1BQK2R w KQkq - 0 1")
# board3 = chess.Board("2r1kb1r/pb1n1pp1/2q1p2p/2pNP3/1p1P4/3B1N2/PP3PPP/R1BQR1K1 b k - 0 15")
# board4 = chess.Board("3qkb1r/1b1n1ppp/2p1pn2/2Pp4/QP1P1B2/4PN2/5PPP/4KB1R b Kk - 0 13")

# board5 = chess.Board("8/4P3/7p/5BpP/2p3P1/1k3b2/2p5/2K5 b - - 0 61")
# board6 = chess.Board("r1bqkb1r/p2n1ppp/2p1pn2/1p6/3P4/1BN1PN2/PP3PPP/R1BQK2R w KQkq - 0 1")
# board7 = chess.Board("2r1kb1r/pb1n1pp1/2q1p2p/2pNP3/1p1P4/3B1N2/PP3PPP/R1BQR1K1 b k - 0 15")
# board8 = chess.Board("3qkb1r/1b1n1ppp/2p1pn2/2Pp4/QP1P1B2/4PN2/5PPP/4KB1R b Kk - 0 13")

# # input_data = np.array([encode_board(board)], dtype=np.float32)

# boards = [board1, board2, board3, board4, board5, board6, board7, board8]

# # # Run ONNX prediction
# # sess = ort.InferenceSession("NNUE_flat_with_phase_21_to_61.onnx")


# # t0 = timer()
# # onnx_output = sess.run(None, {"input": input_data})
# # result = reverse_scaling_and_unclip(onnx_output[0][0][0])
# # t1 = timer()
# # print('ONNX output:', result, t1 - t0)

# t0 = timer()
# for board in boards:    
#     result = evaluate_board(board)
# t1 = timer()
# print('Actual Eval:', result, t1 - t0, (t1 - t0)/ len(boards))

# model = onnx.load("NNUE.onnx")
# print(model.graph.input[0].type.tensor_type.shape)

board = chess.Board("r1r3k1/p4ppp/1p1qpn2/3pn3/8/P3Q3/1P1NBPPP/R1B2RK1 b - - 5 19")
t0 = timer()  
result = evaluate_board(board)
t1 = timer()
print('Actual Eval:', result, t1 - t0)

print()

board = chess.Board("1r1r2k1/p1bn1ppp/B3pn2/8/2NPp2P/P7/1P3PP1/R1B1K2R w KQ - 3 20")
t0 = timer()  
result = evaluate_board(board)
t1 = timer()
print('Actual Eval:', result, t1 - t0)

print()

board = chess.Board("1r1qnrk1/p2nbppp/1p6/1B1p4/3Pp3/P3PP2/1P1NQ1PP/R1B2RK1 b - - 0 19")
t0 = timer()  
result = evaluate_board(board)
t1 = timer()
print('Actual Eval:', result, t1 - t0)