import tensorflow as tf
def dice_loss(y_true, y_pred):
    smooth = 1e-6
    y_true = tf.squeeze(y_true, axis=-1)
    y_true = tf.one_hot(tf.cast(y_true, tf.int32), depth=7)
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def focal_loss(y_true, y_pred, gamma=2., alpha=0.25):
    y_pred = tf.clip_by_value(y_pred, 1e-6, 1 - 1e-6)
    y_true = tf.squeeze(y_true, axis=-1)
    y_true = tf.one_hot(tf.cast(y_true, tf.int32), depth=7)
    return -tf.reduce_sum(alpha * y_true * tf.pow(1 - y_pred, gamma) * tf.math.log(y_pred))

def cat_cross_entropy(y_true, y_pred):
    y_true = tf.squeeze(y_true, axis=-1)
    return tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred))
def combined_loss(y_true, y_pred, alpha=0.33, beta=0.33, gamma=0.34):
    return alpha * dice_loss(y_true, y_pred) + beta * focal_loss(y_true, y_pred) + gamma * cat_cross_entropy(y_true, y_pred)

