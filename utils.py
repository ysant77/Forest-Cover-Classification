import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K


EPSILON = 1e-7
SAMPLE_WEIGHT = [EPSILON, 1.1033, 1.0629, 1.1714, 1.1273, 1.2035, 1.0667]
weights = np.array(SAMPLE_WEIGHT)

def weighted_categorical_crossentropy(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy

    Variables:
        weights: numpy array of shape (C,) where C is the number of classes
    """
    weights = tf.constant(weights, dtype=tf.float32)
    
    def loss(y_true, y_pred):
        #y_true = tf.cast(y_true, tf.float32)

        y_true = tf.one_hot(tf.squeeze(y_true, axis=-1), depth=7)
        class_loglosses = K.mean(K.categorical_crossentropy(y_true, y_pred, from_logits=False), axis=[0, 1, 2])
        return K.sum(class_loglosses * K.constant(weights))


    return loss

custom_loss = weighted_categorical_crossentropy(weights)
