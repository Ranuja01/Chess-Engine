import tensorflow as tf
import numpy as np

# Load the model
model = tf.keras.models.load_model("NNUE_flat_with_phase_21_to_61.keras")

# Check model summary (for layer names and shapes)
model.summary()

# Directory to save raw binary files
output_dir = "./weights/"
import os
os.makedirs(output_dir, exist_ok=True)

# Loop through layers and export weights
for i, layer in enumerate(model.layers):
    weights = layer.get_weights()
    if len(weights) == 0:
        continue  # Skip layers like InputLayer

    kernel, bias = weights  # Fully connected: (weights, biases)
    print(f"Layer {i}: {layer.name}, kernel: {kernel.shape}, bias: {bias.shape}")

    # Save weights
    kernel.astype(np.float32).tofile(f"{output_dir}/w{i}.bin")  # layer indices start at 1
    bias.astype(np.float32).tofile(f"{output_dir}/b{i}.bin")
