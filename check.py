import tensorflow as tf
import keras

print("TensorFlow version:", tf.__version__)
print("Keras version:", keras.__version__)

# Check if GPU is available
print("\nGPU Available:", tf.config.list_physical_devices('GPU'))