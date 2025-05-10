import tensorflow as tf
import numpy as np
import os

# Load the model
model = tf.keras.models.load_model("NNUE_flat_with_phase_21_to_61.keras")
model.summary()

output_dir = "./weights_quant/"
os.makedirs(output_dir, exist_ok=True)

scales = []

for i, layer in enumerate(model.layers):
    weights = layer.get_weights()
    if len(weights) == 0:
        continue

    kernel, bias = weights  # (weights, biases)

    # Determine scale for this layer
    max_abs_weight = np.max(np.abs(kernel))
    scale = max_abs_weight / 127.0 if max_abs_weight > 0 else 1.0
    scales.append(scale)

    # Quantize weights and biases
    q_kernel = np.round(kernel / scale).astype(np.int8)
    q_bias = np.round(bias / (scale * scale)).astype(np.int32)

    # Save quantized weights and biases
    q_kernel.tofile(f"{output_dir}/w{i}.bin")
    q_bias.tofile(f"{output_dir}/b{i}.bin")

# Save scales as float32 (you'll need them in C++)
np.array(scales, dtype=np.float32).tofile(f"{output_dir}/scales.bin")
