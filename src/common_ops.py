import numpy as np
import tensorflow as tf


def create_weight(name, shape, initializer=None, trainable=True, seed=None):
    if initializer is None:
        initializer = tf.contrib.keras.initializers.he_normal(seed=seed)
    return tf.get_variable(name, shape, initializer=initializer, trainable=trainable)


def create_bias(name, shape, initializer=None):
    if initializer is None:
        initializer = tf.constant_initializer(0.0, dtype=tf.float32)
    return tf.get_variable(name, shape, initializer=initializer)

