import tensorflow as tf
from tensorflow.keras import backend as K
import gc
import os, psutil

tf.compat.v1.keras.backend.clear_session()

# If using TF1:
tf.compat.v1.reset_default_graph()

# Force Python to clean up objects
gc.collect()

# Explicitly kill the current process (optional but effective)
psutil.Process(os.getpid()).terminate()