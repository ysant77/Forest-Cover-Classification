import base64
import os
from io import BytesIO

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
import tifffile
from PIL import Image
from fastapi import HTTPException

if not os.path.exists("files"):
    os.makedirs("files")
if not os.path.exists("analysis"):
    os.makedirs("analysis")

MAX_VAL = 4096.0

root_dir = os.path.dirname(__file__)
model_dir_name = os.path.join(root_dir, "models")

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


def normalize(image: np.ndarray):
    """Normalizes numpy arrays into scale 0.0 - 1.0"""
    # array_min, array_max = array.min(), array.max()
    tf_img = tf.clip_by_value(tf.cast(image, tf.float32) / MAX_VAL, 0., 1.)
    img = tf_img.numpy()

    return img


def numpy_to_base64_rgb(numpy_img):
    """Converts a numpy array to a base64 encoded RGB image."""
    img = Image.fromarray(numpy_img.astype('uint8'), 'RGB')
    buffered = BytesIO()
    img.save(buffered, format="PNG")

    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def loading_tiff_and_resize(img_bytes: bytes, desired_shape: tuple):
    with tifffile.TiffFile(BytesIO(img_bytes)) as tif:
        img = tif.asarray()

    # Check image shape
    img_shape = img.shape

    if img_shape[2] != desired_shape[2]:
        raise HTTPException(status_code=400, detail="Incorrect number of bands in the image.")

    if img_shape[0] < desired_shape[0] or img_shape[1] < desired_shape[1]:
        raise HTTPException(status_code=400, detail="Insufficient image size. Please make sure the image is at least "
                                                    "(256, 256, 13")

    if img_shape[0] != desired_shape[0] or img_shape[1] != desired_shape[1]:
        # Image reshape to (256, 256, 13)
        print(f"Original image shape: {img_shape}")
        img = cv2.resize(img, (desired_shape[0], desired_shape[1]))
        # img = tf.image.resize(img, (desired_shape[0], desired_shape[1])).numpy()
        print(f"Resized image shape: {img.shape}")

    return img
