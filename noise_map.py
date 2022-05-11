import tensorflow as tf
import numpy as np

def get_noise_map(dataset: np.ndarray) -> np.ndarray:
    """Function to generate a noise map to be concatenated to the input data channels.

    Values for the noise map are taken from a uniform distribution.
    """
    noiseIntL = [0, 75]
    noiseIntL[0] /= 255
    noiseIntL[1] /= 255

    stdn = np.random.uniform(noiseIntL[0], noiseIntL[1], \
                    size=[dataset.shape[0], dataset.shape[1], \
                         dataset.shape[2], 1])

    stdn_var = tf.Variable(stdn, dtype=tf.float16);
    return stdn