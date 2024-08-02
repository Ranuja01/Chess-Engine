from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras.regularizers import L2
from tensorflow.keras.layers import (
    Conv2D,
    Flatten,
    Dense,
    InputLayer,
    Dropout,
    BatchNormalization,
    MultiHeadAttention,
)
import numpy as np

# Assuming you have saved and loaded your model
data_path = r"../Models/WhiteModel_21_36(5)_selfplay.keras"

model = tf.keras.models.load_model(data_path)

def check_l2_regularization(model):
    """Check if layers use L2 regularization and print the regularization factor."""
    for layer in model.layers:
        if hasattr(layer, "kernel_regularizer") and isinstance(layer.kernel_regularizer, L2):
            print(f"Layer {layer.name} uses L2 regularization with factor {layer.kernel_regularizer.l2:.5f}")
        else:
            print(f"Layer {layer.name} does not use L2 regularization")


def check_dropout_rate(model):
    """Check if layers use Dropout and print the dropout rate."""
    for layer in model.layers:
        if isinstance(layer, Dropout):
            print(f"Layer {layer.name} uses Dropout with rate {layer.rate:.2f}")
        else:
            print(f"Layer {layer.name} does not use Dropout")


def check_multihead_attention_details(model):
    """Check details of MultiHeadAttention layers and print num_heads and ff_dims."""
    for layer in model.layers:
        if isinstance(layer, MultiHeadAttention):
            num_heads = layer.num_heads
            key_dim = layer.key_dim
            
            # Assume input shapes as (batch_size, sequence_length, feature_dim)
            query_shape = (None, 8*8, 12)  # Adjust these as per your actual input shapes
            key_shape = (None, 8*8, 12)
            value_shape = (None, 8*8, 12)
            
            try:
                # Get the output shape of the layer
                #output_shape = layer.compute_output_shape([query_shape, key_shape, value_shape])[-1]
                
                print(f"Layer {layer.name} uses MultiHeadAttention with {num_heads} heads, key_dim {key_dim}, and output shape")
            except Exception as e:
                print(f"Error obtaining output shape for {layer.name}: {str(e)}")
        else:
            print(f"Layer {layer.name} is not a MultiHeadAttention layer")


# Example usage
print("Checking Dropout Rates:")
check_dropout_rate(model)

print("\nChecking L2 Regularization:")
check_l2_regularization(model)

print("\nChecking MultiHeadAttention Details:")
check_multihead_attention_details(model)
